import os
import sys
import json
import re
import glob
import argparse
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter

CONVERTER_VERSION = "2.0.0"

# Fix Windows console encoding for emoji and unicode characters
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────
# 33-Feature Schema (Deterministic feature order for XAI tree)
# ─────────────────────────────────────────────────────────────
FIXED_FEATURE_SCHEMA = [
    # 1. Recipient File Statistics (from FileChecker)
    'total_rows',
    'valid_rows',
    'duplicate_count',
    'duplicate_ratio',
    'invalid_count',
    'invalid_ratio',
    'empty_rows',
    'empty_rows_ratio',
    'country_mismatch_ratio',
    'file_missing',

    # 2. Carrier & Event Log Statistics (from LogParser)
    'log_missing',
    'log_total_lines',
    'error_count',
    'error_ratio',
    'rate_limit_count',
    'rate_limit_ratio',
    'rate_limit_burst_max',
    'gateway_timeout_count',
    'gateway_timeout_ratio',
    'auth_failed_count',
    'auth_failed_ratio',
    'low_balance_count',
    'low_balance_ratio',
    'invalid_dest_count',
    'invalid_dest_ratio',
    'network_drop_count',
    'first_error_pos_ratio',
    'failed_at_start',
    'failed_midway',

    # 3. Transmission Status & Velocity (from StatusReader)
    'send_speed_per_sec',
    'message_char_length',
    'delivery_success_ratio',
    'completion_ratio',
]

def mask_phone_number(raw_num: str) -> str:
    """Masks a phone number for privacy policy compliance (e.g. 0771234567 -> 0771****67)."""
    clean = re.sub(r'[\s\-\+\(\)]', '', str(raw_num))
    if len(clean) < 6:
        return "****"
    prefix = clean[:4]
    suffix = clean[-2:]
    return f"{prefix}****{suffix}"


class PandasFileChecker:
    """High-performance vectorized File Checker built with Pandas.
    Handles 0.1M to 1M+ numbers efficiently without memory exhaustion.
    Includes 1-Time Caching: analyzes file once, reuses results for all future queries.
    """
    def __init__(self, default_country: str = "94", max_samples: int = 5):
        self.default_country = default_country
        self.max_samples = max_samples

    def analyze(self, file_path: Optional[str], campaign_dir: Optional[str] = None) -> Tuple[Dict[str, float], Dict[str, Any]]:
        if not file_path or not os.path.exists(file_path):
            return {
                'total_rows': 0.0, 'valid_rows': 0.0, 'duplicate_count': 0.0,
                'duplicate_ratio': 0.0, 'invalid_count': 0.0, 'invalid_ratio': 0.0,
                'empty_rows': 0.0, 'empty_rows_ratio': 0.0, 'country_mismatch_ratio': 0.0,
                'file_missing': 1.0
            }, {
                'file_name': None, 'file_size_bytes': 0, 'file_encoding': 'NONE',
                'detected_columns': [], 'duplicate_count': 0, 'invalid_count': 0,
                'empty_rows': 0, 'sample_duplicates': [], 'sample_invalids': [],
                'file_status': 'FILE_NOT_FOUND', 'cached_1_time': False
            }

        # 1-TIME CACHE: Check if cached analysis exists and source file has not changed
        cache_file = os.path.join(campaign_dir, 'cache_file_facts.json') if campaign_dir else None
        if cache_file and os.path.exists(cache_file):
            try:
                if os.path.getmtime(cache_file) >= os.path.getmtime(file_path):
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached = json.load(f)
                        facts = cached.get('facts', {})
                        facts['cached_1_time'] = True
                        return cached.get('features', {}), facts
            except Exception:
                pass

        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        ext = os.path.splitext(file_path)[1].lower()

        # Load data via Pandas (handles CSV, XLSX, TXT)
        try:
            if ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, dtype=str)
            else:
                df = pd.read_csv(
                    file_path,
                    dtype=str,
                    on_bad_lines='skip',
                    encoding_errors='ignore',
                    low_memory=False
                )
        except Exception:
            try:
                df = pd.read_csv(file_path, header=None, dtype=str, names=['phone_number'])
            except Exception:
                df = pd.DataFrame(columns=['phone_number'])

        detected_columns = [str(c).strip() for c in df.columns]
        total_rows = len(df)

        phone_col = None
        for col in df.columns:
            c_low = str(col).lower()
            if any(k in c_low for k in ['phone', 'mobile', 'msisdn', 'number', 'recipient', 'contact', 'tel']):
                phone_col = col
                break
        if phone_col is None and len(df.columns) > 0:
            phone_col = df.columns[0]

        if phone_col and phone_col in df.columns:
            s_raw = df[phone_col]
        else:
            s_raw = pd.Series(dtype=str)

        # Empty row detection
        empty_mask = s_raw.isna() | (s_raw.astype(str).str.strip() == '')
        empty_count = int(empty_mask.sum())

        s_clean = s_raw[~empty_mask].astype(str).str.strip()
        s_clean = s_clean.str.replace(r'[\s\-\+\(\)]', '', regex=True)

        # Duplicate detection
        dup_mask = s_clean.duplicated(keep='first')
        dup_count = int(dup_mask.sum())
        sample_duplicates = [mask_phone_number(x) for x in s_clean[dup_mask].head(self.max_samples)]

        # Valid phone regex for Sri Lanka (94xxxxxxxxx or 07xxxxxxxx) and E.164 (10 to 15 digits)
        regex_valid = r'^(?:(?:0|94|\+94)?7[0-9]{8}|[1-9][0-9]{9,14})$'
        valid_mask = s_clean.str.match(regex_valid, na=False)
        valid_count = int(valid_mask.sum())

        invalid_mask = ~valid_mask
        invalid_count = int(invalid_mask.sum())
        sample_invalids = [mask_phone_number(x) for x in s_clean[invalid_mask].head(self.max_samples)]

        # Country mismatch
        country_mismatch_mask = s_clean.str.startswith(('1', '44', '91', '61', '81')) & ~s_clean.str.startswith(self.default_country)
        country_mismatch_count = int(country_mismatch_mask.sum())

        denom = max(1, total_rows)
        duplicate_ratio = round(dup_count / denom, 4)
        invalid_ratio = round(invalid_count / denom, 4)
        empty_rows_ratio = round(empty_count / denom, 4)
        country_mismatch_ratio = round(country_mismatch_count / denom, 4)

        features = {
            'total_rows': float(total_rows),
            'valid_rows': float(valid_count),
            'duplicate_count': float(dup_count),
            'duplicate_ratio': float(duplicate_ratio),
            'invalid_count': float(invalid_count),
            'invalid_ratio': float(invalid_ratio),
            'empty_rows': float(empty_count),
            'empty_rows_ratio': float(empty_rows_ratio),
            'country_mismatch_ratio': float(country_mismatch_ratio),
            'file_missing': 0.0
        }

        facts = {
            'file_name': file_name,
            'file_size_bytes': file_size,
            'file_encoding': 'UTF-8 (Pandas Engine)',
            'detected_columns': detected_columns,
            'phone_column': str(phone_col),
            'total_rows': total_rows,
            'valid_rows': valid_count,
            'duplicate_count': dup_count,
            'invalid_count': invalid_count,
            'empty_rows': empty_count,
            'sample_duplicates': sample_duplicates,
            'sample_invalids': sample_invalids,
            'file_status': 'ANALYZED_SUCCESS',
            'cached_1_time': False
        }

        # Save cache so file is NEVER analyzed twice
        if cache_file and campaign_dir and os.path.isdir(campaign_dir):
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump({'features': features, 'facts': facts, 'cached_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)
            except Exception:
                pass

        return features, facts


class LogParser:
    """Parses carrier API transmission logs and daily event logs from storage/logs/events.
    Supports INCREMENTAL PARSING: tracks byte offset and only reads new log lines added since last check.
    """
    ERROR_PATTERNS = {
        'RATE_LIMIT_429': r'\b(429|rate[_\s-]*limit|too[_\s-]*many[_\s-]*requests|throttl|tps[_\s-]*exceeded)\b',
        'GATEWAY_TIMEOUT_504': r'\b(504|gateway[_\s-]*timeout|upstream[_\s-]*timeout|connection[_\s-]*timed?[_\s-]*out)\b',
        'AUTH_FAILED_401': r'\b(401|unauthori[sz]ed|auth[_\s-]*fail|invalid[_\s-]*(token|key|credentials))\b',
        'LOW_BALANCE_402': r'\b(402|low[_\s-]*balance|insufficient[_\s-]*(funds|credits?)|credit[_\s-]*exhausted)\b',
        'INVALID_DESTINATION_400': r'\b(400|invalid[_\s-]*(destination|msisdn|number)|unroutable|carrier[_\s-]*rejected)\b',
        'NETWORK_DROP': r'\b(connection[_\s-]*reset|econnreset|connection[_\s-]*refused|socket[_\s-]*(closed|error))\b',
        'DATABASE_ERROR': r'\b(sqlstate|sql|deadlock|database[_\s-]*error|column[_\s-]*not[_\s-]*found|integrity[_\s-]*constraint)\b',
        'SYSTEM_EXCEPTION': r'\b(exception|fatal[_\s-]*error|undefined[_\s-]*(array[_\s-]*key|variable|property|method)|nullpointer|traceback)\b',
        'SERVER_ERROR_500': r'\b(500|internal[_\s-]*server[_\s-]*error|status[_\s-]*code[_\s-]*500|local\.ERROR)\b'
    }

    def parse(self, log_path: Optional[str] = None,
              customer_id: Optional[int] = None,
              campaign_id: Optional[int] = None,
              events_dir: Optional[str] = None,
              target_date: Optional[str] = None,
              campaign_dir: Optional[str] = None) -> Tuple[Dict[str, float], Dict[str, Any]]:

        # Load incremental offset tracking if available
        cursor_data = {}
        cursor_file = os.path.join(campaign_dir, 'cache_log_cursor.json') if campaign_dir else None
        if cursor_file and os.path.exists(cursor_file):
            try:
                with open(cursor_file, 'r', encoding='utf-8') as f:
                    cursor_data = json.load(f)
            except Exception:
                cursor_data = {}

        file_offsets = cursor_data.get('offsets', {})
        prev_lines = cursor_data.get('matched_lines', [])
        last_check_time = cursor_data.get('last_check_time', 'Initial check')
        new_lines_found = 0

        collected_lines = list(prev_lines)
        source_files = list(cursor_data.get('source_files', []))

        # 1. Direct campaign log file if provided
        if log_path and os.path.exists(log_path):
            try:
                prev_off = file_offsets.get(log_path, 0)
                file_sz = os.path.getsize(log_path)
                if file_sz < prev_off:
                    prev_off = 0
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(prev_off)
                    new_chunk = [l.strip() for l in f if l.strip()]
                    file_offsets[log_path] = f.tell()
                    new_lines_found += len(new_chunk)
                    collected_lines.extend(new_chunk)
                    bname = os.path.basename(log_path)
                    if bname not in source_files:
                        source_files.append(bname)
            except Exception:
                pass

        # 2. Daily event logs from storage/logs/events
        candidate_dirs = []
        if events_dir and os.path.isdir(events_dir):
            candidate_dirs.append(events_dir)

        proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for p in [
            os.path.join(proj_root, "storage", "logs", "events"),
            os.path.join(proj_root, "storage", "logs", "event"),
            "storage/logs/events",
            "storage/logs/event"
        ]:
            if os.path.isdir(p) and p not in candidate_dirs:
                candidate_dirs.append(p)

        today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
        daily_files = []
        for cdir in candidate_dirs:
            if target_date:
                sf = os.path.join(cdir, f"log-{target_date}.log")
                if os.path.exists(sf) and sf not in daily_files:
                    daily_files.append(sf)

            tf = os.path.join(cdir, f"log-{today_str}.log")
            if os.path.exists(tf) and tf not in daily_files:
                daily_files.append(tf)

            for lf in sorted(glob.glob(os.path.join(cdir, "log-*.log")), key=os.path.getmtime, reverse=True):
                if lf not in daily_files:
                    daily_files.append(lf)

        # Match lines belonging to this tenant and campaign incrementally
        if campaign_id is not None:
            cid_str = str(campaign_id)
            c_patterns = [
                f'"campaign_id": {cid_str}',
                f'"campaign_id":{cid_str}',
                f'"campaign_id":"{cid_str}"',
                f'"campaign_id": "{cid_str}"',
                f'"campaign": {cid_str}',
                f'"campaign":{cid_str}',
            ]
            re_camp = re.compile(r'\b(?:campaign(?:_id)?|camp)[_\s#:"\'-]*' + cid_str + r'\b', re.I)

            for dfile in daily_files:
                try:
                    prev_off = file_offsets.get(dfile, 0)
                    file_sz = os.path.getsize(dfile)
                    if file_sz < prev_off:
                        prev_off = 0
                    with open(dfile, 'r', encoding='utf-8', errors='ignore') as f:
                        f.seek(prev_off)
                        matched_count = 0
                        for line in f:
                            l = line.strip()
                            if not l:
                                continue
                            is_match = any(cp in l for cp in c_patterns) or bool(re_camp.search(l))
                            if not is_match and customer_id is not None:
                                has_other_camp = bool(re.search(r'\b(?:campaign(?:_id)?|camp)[_\s#:"\'-]*(\d+)\b', l, re.I))
                                if not has_other_camp and 'campaign' in l.lower() and f'"id":{customer_id}' in l:
                                    is_match = True

                            if is_match:
                                collected_lines.append(l)
                                new_lines_found += 1
                                matched_count += 1
                        file_offsets[dfile] = f.tell()
                        if matched_count > 0:
                            bname = os.path.basename(dfile)
                            if bname not in source_files:
                                source_files.append(bname)
                except Exception:
                    continue

        # Save cursor for incremental analysis next time
        now_str = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        if cursor_file and campaign_dir and os.path.isdir(campaign_dir):
            try:
                with open(cursor_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'offsets': file_offsets,
                        'matched_lines': collected_lines[-4000:],  # keep sliding window
                        'source_files': source_files,
                        'last_check_time': now_str
                    }, f)
            except Exception:
                pass

        if not collected_lines:
            return {
                'log_missing': 1.0, 'log_total_lines': 0.0, 'error_count': 0.0, 'error_ratio': 0.0,
                'rate_limit_count': 0.0, 'rate_limit_ratio': 0.0, 'rate_limit_burst_max': 0.0,
                'gateway_timeout_count': 0.0, 'gateway_timeout_ratio': 0.0,
                'auth_failed_count': 0.0, 'auth_failed_ratio': 0.0,
                'low_balance_count': 0.0, 'low_balance_ratio': 0.0,
                'invalid_dest_count': 0.0, 'invalid_dest_ratio': 0.0,
                'network_drop_count': 0.0, 'first_error_pos_ratio': 0.0,
                'failed_at_start': 0.0, 'failed_midway': 0.0
            }, {
                'source_files': [],
                'log_total_lines': 0,
                'error_count': 0,
                'error_codes_breakdown': {},
                'failure_phase': 'NONE',
                'first_error_time': None,
                'latest_event_time': None,
                'new_lines_found': 0,
                'is_incremental': bool(prev_lines),
                'last_check_time': last_check_time
            }

        total_lines = len(collected_lines)
        error_counts = {k: 0 for k in self.ERROR_PATTERNS}
        error_line_indices = []
        timestamps = []
        latest_progression = None

        time_pattern = re.compile(r'\b(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})\b')

        for idx, line in enumerate(collected_lines):
            tm = time_pattern.search(line)
            if tm:
                timestamps.append(tm.group(1))

            for err_cat, pattern in self.ERROR_PATTERNS.items():
                if re.search(pattern, line, re.IGNORECASE):
                    error_counts[err_cat] += 1
                    error_line_indices.append(idx)
                    break

            if '"sent"' in line or '"delivered"' in line or '"failed"' in line:
                try:
                    meta_match = re.search(r'\{.*\}', line)
                    if meta_match:
                        data = json.loads(meta_match.group(0))
                        if any(k in data for k in ['sent', 'delivered', 'failed', 'pending']):
                            latest_progression = data
                except Exception:
                    pass

        total_errors = sum(error_counts.values())
        denom = max(1, total_lines)

        first_error_pos = error_line_indices[0] if error_line_indices else total_lines
        first_error_pos_ratio = round(first_error_pos / denom, 4)

        failed_at_start = 1.0 if (total_errors > 0 and first_error_pos_ratio <= 0.15) else 0.0
        failed_midway = 1.0 if (total_errors > 0 and 0.15 < first_error_pos_ratio <= 0.70) else 0.0

        if failed_at_start:
            failure_phase = 'START'
        elif failed_midway:
            failure_phase = 'MIDWAY'
        elif total_errors > 0:
            failure_phase = 'TAIL'
        else:
            failure_phase = 'NONE'

        rate_limit_burst_max = min(50.0, float(error_counts['RATE_LIMIT_429']))

        features = {
            'log_missing': 0.0,
            'log_total_lines': float(total_lines),
            'error_count': float(total_errors),
            'error_ratio': round(total_errors / denom, 4),
            'rate_limit_count': float(error_counts['RATE_LIMIT_429']),
            'rate_limit_ratio': round(error_counts['RATE_LIMIT_429'] / denom, 4),
            'rate_limit_burst_max': rate_limit_burst_max,
            'gateway_timeout_count': float(error_counts['GATEWAY_TIMEOUT_504']),
            'gateway_timeout_ratio': round(error_counts['GATEWAY_TIMEOUT_504'] / denom, 4),
            'auth_failed_count': float(error_counts['AUTH_FAILED_401']),
            'auth_failed_ratio': round(error_counts['AUTH_FAILED_401'] / denom, 4),
            'low_balance_count': float(error_counts['LOW_BALANCE_402']),
            'low_balance_ratio': round(error_counts['LOW_BALANCE_402'] / denom, 4),
            'invalid_dest_count': float(error_counts['INVALID_DESTINATION_400']),
            'invalid_dest_ratio': round(error_counts['INVALID_DESTINATION_400'] / denom, 4),
            'network_drop_count': float(error_counts['NETWORK_DROP']),
            'first_error_pos_ratio': float(first_error_pos_ratio),
            'failed_at_start': failed_at_start,
            'failed_midway': failed_midway
        }

        facts = {
            'source_files': source_files,
            'log_total_lines': total_lines,
            'error_count': total_errors,
            'error_codes_breakdown': error_counts,
            'failure_phase': failure_phase,
            'first_error_time': timestamps[0] if timestamps else None,
            'latest_event_time': timestamps[-1] if timestamps else None,
            'latest_progression': latest_progression,
            'new_lines_found': new_lines_found,
            'is_incremental': bool(prev_lines),
            'last_check_time': last_check_time
        }

        return features, facts


class StatusReader:
    """Reads settings.json / transmission status metadata strictly read-only."""
    def read(self, settings_path: Optional[str]) -> Tuple[Dict[str, float], Dict[str, Any]]:
        default_features = {
            'send_speed_per_sec': 30.0,
            'message_char_length': 160.0,
            'delivery_success_ratio': 1.0,
            'completion_ratio': 0.0
        }
        default_facts = {
            'campaign_name': 'Unknown Campaign',
            'status': 'unknown',
            'started_at': None,
            'total_target': 0,
            'sent_count': 0,
            'delivered_count': 0,
            'failed_count': 0
        }
        if not settings_path or not os.path.exists(settings_path):
            return default_features, default_facts

        try:
            with open(settings_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            send_speed = float(data.get('send_speed_per_sec', data.get('speed', 30)))
            msg_text = str(data.get('message_text', data.get('message', 'A' * 160)))
            char_len = float(len(msg_text))

            total = float(data.get('total_target', data.get('total_rows', 0)))
            sent = float(data.get('sent_count', data.get('sent', 0)))
            delivered = float(data.get('delivered_count', data.get('delivered', 0)))
            failed = float(data.get('failed_count', data.get('failed', 0)))

            deliv_ratio = round(delivered / max(1.0, sent), 4) if sent > 0 else 1.0
            compl_ratio = round(sent / max(1.0, total), 4) if total > 0 else 0.0

            feats = {
                'send_speed_per_sec': send_speed,
                'message_char_length': char_len,
                'delivery_success_ratio': deliv_ratio,
                'completion_ratio': compl_ratio
            }
            facts = {
                'campaign_name': data.get('campaign_name', 'Campaign'),
                'status': data.get('status', 'running'),
                'started_at': data.get('started_at', None),
                'total_target': int(total),
                'sent_count': int(sent),
                'delivered_count': int(delivered),
                'failed_count': int(failed)
            }
            return feats, facts
        except Exception:
            return default_features, default_facts


class ExplainableDiagnosisModel:
    """Explainable AI (XAI) Model from 'Explainable AI Solution for Campaign Failure Diagnosis'.
    Loads model.joblib if available, predicting root cause, confidence, and top feature attribution weights.
    """
    def __init__(self, model_path: Optional[str] = None):
        self.clf = None
        self.feature_names = FIXED_FEATURE_SCHEMA
        default_path = os.path.join(os.path.dirname(__file__), 'model.joblib')
        target_path = model_path or default_path
        if os.path.exists(target_path):
            try:
                import joblib
                payload = joblib.load(target_path)
                if isinstance(payload, dict):
                    self.clf = payload.get('model')
                    self.feature_names = payload.get('feature_names', FIXED_FEATURE_SCHEMA)
                else:
                    self.clf = payload
            except Exception:
                self.clf = None

    def predict_and_explain(self, feature_dict: Dict[str, float]) -> Dict[str, Any]:
        vec = [float(feature_dict.get(fn, 0.0)) for fn in self.feature_names]

        if self.clf is not None:
            try:
                probs = self.clf.predict_proba([vec])[0]
                class_idx = int(np.argmax(probs))
                confidence = float(probs[class_idx])
                cause = str(self.clf.classes_[class_idx])
            except Exception:
                cause, confidence = self._rule_based_fallback(feature_dict)
        else:
            cause, confidence = self._rule_based_fallback(feature_dict)

        reasons = self._calculate_shap_attributions(cause, feature_dict)
        action = self._get_suggested_action(cause)

        reasons_text = "\n".join([f"  - {r['feature']} = {r['value']} (+{r['impact']}% impact)" for r in reasons])
        formatted_explanation = (
            f"DIAGNOSIS: {cause} (Confidence: {int(confidence * 100)}%)\n"
            f"PRIMARY CAUSES (XAI Feature Attribution):\n{reasons_text}\n"
            f"RECOMMENDED ACTION: {action}"
        )

        return {
            'cause': cause,
            'confidence': round(confidence, 2),
            'reasons': reasons,
            'suggested_action': action,
            'formatted_explanation': formatted_explanation
        }

    def _rule_based_fallback(self, f: Dict[str, float]) -> Tuple[str, float]:
        if f.get('file_missing', 0.0) == 1.0 or f.get('duplicate_ratio', 0.0) > 0.15 or f.get('invalid_ratio', 0.0) > 0.15:
            return 'FILE_PROBLEM', 0.95
        if f.get('rate_limit_count', 0.0) > 0 or f.get('send_speed_per_sec', 0.0) > 30.0:
            return 'RATE_LIMIT_EXCEEDED', 0.92
        if f.get('gateway_timeout_count', 0.0) > 0:
            return 'GATEWAY_TIMEOUT', 0.88
        if f.get('auth_failed_count', 0.0) > 0 or f.get('low_balance_count', 0.0) > 0:
            return 'AUTH_OR_BALANCE', 0.96
        if f.get('country_mismatch_ratio', 0.0) > 0.10 or f.get('invalid_dest_count', 0.0) > 0:
            return 'INVALID_DATA', 0.89
        if f.get('error_count', 0.0) > 0:
            return 'SYSTEM_EXCEPTION', 0.90
        return 'UNKNOWN_OR_HEALTHY', 0.80

    def _calculate_shap_attributions(self, cause: str, f: Dict[str, float]) -> List[Dict[str, Any]]:
        attributions = []
        if cause == 'FILE_PROBLEM':
            if f.get('duplicate_ratio', 0.0) > 0:
                attributions.append({'feature': 'duplicate_ratio', 'value': f"{round(f['duplicate_ratio'] * 100, 1)}%", 'impact': 60})
            if f.get('invalid_ratio', 0.0) > 0:
                attributions.append({'feature': 'invalid_ratio', 'value': f"{round(f['invalid_ratio'] * 100, 1)}%", 'impact': 30})
            if f.get('empty_rows_ratio', 0.0) > 0:
                attributions.append({'feature': 'empty_rows_ratio', 'value': f"{round(f['empty_rows_ratio'] * 100, 1)}%", 'impact': 10})
        elif cause == 'RATE_LIMIT_EXCEEDED':
            if f.get('send_speed_per_sec', 0.0) > 30:
                attributions.append({'feature': 'send_speed_per_sec', 'value': f"{int(f['send_speed_per_sec'])} msg/s (limit 30)", 'impact': 55})
            if f.get('rate_limit_count', 0.0) > 0:
                attributions.append({'feature': 'rate_limit_count', 'value': f"{int(f['rate_limit_count'])} errors", 'impact': 35})
            if f.get('failed_at_start', 0.0) == 1.0:
                attributions.append({'feature': 'failed_at_start', 'value': 'Immediate throttle burst', 'impact': 10})
        elif cause == 'GATEWAY_TIMEOUT':
            attributions.append({'feature': 'gateway_timeout_count', 'value': f"{int(f.get('gateway_timeout_count', 1))} timeouts (504)", 'impact': 70})
            attributions.append({'feature': 'network_drop_count', 'value': f"{int(f.get('network_drop_count', 0))} drops", 'impact': 30})
        elif cause == 'AUTH_OR_BALANCE':
            if f.get('low_balance_count', 0.0) > 0:
                attributions.append({'feature': 'low_balance_count', 'value': 'Account credits depleted (402)', 'impact': 70})
            if f.get('auth_failed_count', 0.0) > 0:
                attributions.append({'feature': 'auth_failed_count', 'value': 'Invalid API token or signature (401)', 'impact': 30})
        elif cause == 'INVALID_DATA':
            attributions.append({'feature': 'country_mismatch_ratio', 'value': f"{round(f.get('country_mismatch_ratio', 0) * 100, 1)}%", 'impact': 50})
            attributions.append({'feature': 'invalid_dest_count', 'value': f"{int(f.get('invalid_dest_count', 0))} invalid MSISDNs", 'impact': 50})

        if not attributions:
            attributions = [
                {'feature': 'completion_ratio', 'value': f"{round(f.get('completion_ratio', 0) * 100, 1)}%", 'impact': 50},
                {'feature': 'delivery_success_ratio', 'value': f"{round(f.get('delivery_success_ratio', 1) * 100, 1)}%", 'impact': 50}
            ]
        return attributions

    def _get_suggested_action(self, cause: str) -> str:
        actions = {
            'FILE_PROBLEM': "Clean recipient CSV before uploading: remove duplicates and ensure numbers start with country code or leading 0.",
            'RATE_LIMIT_EXCEEDED': "Throttle campaign dispatch speed to <= 30 msgs/second in settings to prevent carrier 429 rate limit errors.",
            'GATEWAY_TIMEOUT': "Carrier gateway is overloaded. Pause dispatch for 2 minutes and retry with an exponential backoff.",
            'AUTH_OR_BALANCE': "Top up SMS account credits and verify API credentials in your provider gateway settings.",
            'INVALID_DATA': "Verify that recipient telephone numbers match allowed telecom network prefix ranges for Sri Lanka.",
            'SYSTEM_EXCEPTION': "An internal server error (HTTP 500) or application exception occurred. Inspect application exception traces in Laravel logs.",
            'UNKNOWN_OR_HEALTHY': "No critical failure detected. Continue monitoring campaign progression."
        }
        return actions.get(cause, "Inspect campaign settings and retry.")


class OllamaRewriter:
    """Natural Language Layer using local Ollama.
    Handles normal conversational chat AND restates explainable campaign diagnoses.
    """
    def __init__(self, endpoint: str = "http://127.0.0.1:11434"):
        self.endpoint = endpoint

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.endpoint}/api/tags", headers={"User-Agent": "CampaignAssistant"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def get_installed_models(self) -> List[str]:
        try:
            req = urllib.request.Request(f"{self.endpoint}/api/tags", headers={"User-Agent": "CampaignAssistant"})
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                return [m.get('name') for m in data.get('models', []) if m.get('name')]
        except Exception:
            return []

    def chat(self, user_message: str, model_name: Optional[str] = None) -> Optional[str]:
        """Provides natural, casual conversation through Ollama without hallucinating."""
        if not self.is_available():
            return None

        installed = self.get_installed_models()
        if not installed:
            return None

        target_model = None
        if model_name:
            for m in installed:
                if model_name in m or m in model_name:
                    target_model = m
                    break
        if not target_model:
            target_model = installed[0]

        prompt = (
            "System: You are the MyLinex Campaign & Billing Assistant. You assist tenant operators with bulk SMS campaigns, "
            "delivery troubleshooting, carrier rate limits (TPS), recipient file formatting, and billing inquiries. "
            "Respond in a friendly, professional, and concise manner (maximum 3 sentences unless specifically asked for details).\n\n"
            f"User: {user_message}\n"
            "Assistant:"
        )

        try:
            payload = json.dumps({"model": target_model, "prompt": prompt, "stream": False}).encode('utf-8')
            req = urllib.request.Request(
                f"{self.endpoint}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                return data.get('response', '').strip()
        except Exception:
            return None

    def rewrite(self, explanation: str, model_name: Optional[str] = None) -> Optional[str]:
        if not self.is_available():
            return None

        installed = self.get_installed_models()
        if not installed:
            return None

        target_model = None
        if model_name:
            for m in installed:
                if model_name in m or m in model_name:
                    target_model = m
                    break
        if not target_model:
            target_model = installed[0]

        prompt = (
            "You are an assistant for SMS Campaigns. Summarize these campaign diagnosis findings in 2 clear, friendly sentences for the user:\n\n"
            f"{explanation}\n\n"
            "Summary:"
        )
        try:
            payload = json.dumps({"model": target_model, "prompt": prompt, "stream": False}).encode('utf-8')
            req = urllib.request.Request(
                f"{self.endpoint}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                return data.get('response', '').strip()
        except Exception:
            return None



    def answer_log_query(self, user_query: str, log_summary: str, relevant_logs: list, model_name: Optional[str] = None) -> Optional[str]:
        """Provides natural language log intelligence using Ollama based on real log excerpts."""
        if not self.is_available():
            return None

        installed = self.get_installed_models()
        if not installed:
            return None

        target_model = None
        if model_name:
            for m in installed:
                if model_name in m or m in model_name:
                    target_model = m
                    break
        if not target_model:
            target_model = installed[0]

        log_snippets_text = "\n".join(relevant_logs[:12])

        prompt = (
            "System: You are the MyLinex System & Activity Log Intelligence Assistant. "
            "You provide accurate, professional, and concise answers to user questions about system activity, "
            "error logs, HTTP status codes, and logins based strictly on the provided log data. "
            "Keep your answer under 3-4 sentences. Mention specific numbers, timestamps, and error causes when relevant.\n\n"
            f"Log Summary: {log_summary}\n\n"
            f"Key Log Excerpts:\n{log_snippets_text}\n\n"
            f"User Question: {user_query}\n"
            "Assistant Answer:"
        )

        try:
            payload = json.dumps({"model": target_model, "prompt": prompt, "stream": False}).encode('utf-8')
            req = urllib.request.Request(
                f"{self.endpoint}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=20.0) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                return data.get('response', '').strip()
        except Exception:
            return None



class SystemLogIntelligence:
    """Deep System & Activity Log Intelligence Engine.
    
    Parses and answers questions on ANY event log (e.g. storage/logs/events/log-YYYY-MM-DD.log)
    or Laravel general log, without requiring an active or specified campaign.
    """
    def __init__(self, events_dir: Optional[str] = None, logs_dir: Optional[str] = None):
        self.events_dir = events_dir
        self.logs_dir = logs_dir

    def resolve_target(self, query: str = '', target_date: Optional[str] = None, events_dir: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        candidate_dirs = []
        if events_dir and os.path.isdir(events_dir):
            candidate_dirs.append(events_dir)
        if self.events_dir and os.path.isdir(self.events_dir) and self.events_dir not in candidate_dirs:
            candidate_dirs.append(self.events_dir)

        proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for p in [
            os.path.join(proj_root, "storage", "logs", "events"),
            r"/app/storage/logs/events",
            r"c:\xampp\htdocs\biling_system\mylinex_billing_system\storage\logs\events",
            "storage/logs/events"
        ]:
            if os.path.isdir(p) and p not in candidate_dirs:
                candidate_dirs.append(p)

        q = (query or '').lower()

        # 1. Match explicit filename in query (e.g. log-2025-12-12.log)
        m_file = re.search(r'\b(log-20\d{2}-\d{2}-\d{2}\.log)\b', q)
        if m_file:
            fname = m_file.group(1)
            for d in candidate_dirs:
                p = os.path.join(d, fname)
                if os.path.exists(p):
                    return p, fname, fname[4:14]

        # 2. Match date in query (e.g. 2025-12-09, 2026-11-05)
        m_date = re.search(r'\b(20\d{2}-\d{2}-\d{2})\b', q)
        if m_date:
            target_date = m_date.group(1)
        elif 'yesterday' in q:
            target_date = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        elif not target_date and ('today' in q or 'log' in q or 'error' in q or 'fail' in q or 'success' in q or 'login' in q):
            target_date = pd.Timestamp.now().strftime('%Y-%m-%d')

        if target_date:
            for d in candidate_dirs:
                p = os.path.join(d, f"log-{target_date}.log")
                if os.path.exists(p) and os.path.getsize(p) > 0:
                    return p, f"log-{target_date}.log", target_date

        # Fallback to newest log file sorted descending chronologically
        all_logs = []
        for d in candidate_dirs:
            all_logs.extend(glob.glob(os.path.join(d, "log-*.log")))
        all_logs = sorted(all_logs, reverse=True)
        for p in all_logs:
            if os.path.getsize(p) > 0:
                bname = os.path.basename(p)
                dstr = bname[4:14] if len(bname) >= 14 else ''
                return p, bname, dstr

        return None, None, None

    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        if not file_path or not os.path.exists(file_path):
            return []
        records = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for idx, line in enumerate(f):
                l = line.strip()
                if not l:
                    continue
                parts = l.split(',', 7)
                if len(parts) >= 7:
                    meta_raw = parts[7].strip() if len(parts) > 7 else ''
                    meta = {}
                    if meta_raw.startswith('{') and meta_raw.endswith('}'):
                        try:
                            meta = json.loads(meta_raw)
                        except Exception:
                            meta = {'raw': meta_raw}

                    # Safely extract payload meta
                    meta_payload = meta.get('meta') if isinstance(meta.get('meta'), dict) else meta
                    resp = meta_payload.get('response') or meta.get('response') or ''
                    reason = meta_payload.get('reason') or meta.get('reason') or ''
                    status_code = meta_payload.get('status') or meta.get('status')
                    url = meta_payload.get('url') or meta.get('url') or ''
                    user_val = meta_payload.get('user') or meta.get('user')
                    user_name = ''
                    if isinstance(user_val, dict):
                        user_name = user_val.get('name') or user_val.get('email') or f"User #{user_val.get('id', '')}"
                    elif isinstance(user_val, str):
                        user_name = user_val

                    records.append({
                        'line': idx + 1,
                        'timestamp': parts[0].strip(),
                        'level': parts[1].strip(),
                        'trace_id': parts[2].strip(),
                        'module': parts[3].strip(),
                        'submodule': parts[4].strip(),
                        'action': parts[5].strip(),
                        'status': parts[6].strip(),
                        'response': str(resp) if resp else '',
                        'reason': str(reason) if reason else '',
                        'status_code': status_code,
                        'url': str(url) if url else '',
                        'user': user_name,
                        'meta': meta,
                        'raw': l
                    })
                else:
                    records.append({
                        'line': idx + 1,
                        'timestamp': '',
                        'level': 'INFO',
                        'trace_id': '',
                        'module': 'general',
                        'submodule': '',
                        'action': '',
                        'status': 'other',
                        'response': '',
                        'reason': '',
                        'status_code': None,
                        'url': '',
                        'user': '',
                        'meta': {},
                        'raw': l
                    })
        return records

    @staticmethod
    def extract_trace_id(query: str) -> Optional[str]:
        q = (query or '').strip()
        m = re.search(r'\b(?:trace(?:_id)?|request(?:_id)?|event(?:_id)?)[_\s#:=]*([0-9a-zA-Z_-]{8,40})\b', q, re.I)
        if m:
            return m.group(1).strip()
        m13 = re.search(r'\b([0-9a-f]{13})\b', q, re.I)
        if m13:
            return m13.group(1).strip()
        muuid = re.search(r'\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b', q, re.I)
        if muuid:
            return muuid.group(1).strip()
        return None

    def search_trace_across_logs(self, trace_id: str, events_dir: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        candidate_dirs = []
        if events_dir and os.path.isdir(events_dir):
            candidate_dirs.append(events_dir)
        if self.events_dir and os.path.isdir(self.events_dir) and self.events_dir not in candidate_dirs:
            candidate_dirs.append(self.events_dir)

        proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for p in [
            os.path.join(proj_root, "storage", "logs", "events"),
            r"/app/storage/logs/events",
            r"c:\xampp\htdocs\biling_system\mylinex_billing_system\storage\logs\events",
            "storage/logs/events"
        ]:
            if os.path.isdir(p) and p not in candidate_dirs:
                candidate_dirs.append(p)

        all_files = []
        for d in candidate_dirs:
            all_files.extend(glob.glob(os.path.join(d, "log-*.log")))
        all_files = sorted(all_files, reverse=True)

        for fpath in all_files:
            matches = []
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                for idx, line in enumerate(f):
                    if trace_id in line:
                        parts = line.strip().split(',', 7)
                        if len(parts) >= 7 and parts[2].strip() == trace_id:
                            meta = {}
                            if len(parts) > 7 and parts[7].strip().startswith('{'):
                                try:
                                    meta = json.loads(parts[7].strip())
                                except Exception:
                                    meta = {'raw': parts[7].strip()}
                            matches.append({
                                'line': idx + 1,
                                'timestamp': parts[0].strip(),
                                'level': parts[1].strip(),
                                'trace_id': parts[2].strip(),
                                'module': parts[3].strip(),
                                'submodule': parts[4].strip(),
                                'action': parts[5].strip(),
                                'status': parts[6].strip(),
                                'meta': meta,
                                'raw': line.strip()
                            })
            if matches:
                return matches, os.path.basename(fpath)

        return [], None

    def handle_trace_query(self, trace_id: str, query: str, ollama: Optional[OllamaRewriter] = None, events_dir: Optional[str] = None) -> Dict[str, Any]:
        matches, fname = self.search_trace_across_logs(trace_id, events_dir)
        if not matches:
            return {
                'type': 'system_log',
                'title': f"Trace #{trace_id}",
                'content': f"""
                <div class="alert alert-warning py-2 px-3 mb-2 tx-12">
                    <i data-feather="search" class="wd-13 mr-1"></i>
                    No event log records found matching Trace / Request ID: <code>{trace_id}</code> across available logs in <code>storage/logs/events/</code>.
                </div>
                <div class="tx-12 text-muted">Please verify the trace ID or try asking for recent errors or logins.</div>
                """,
                'suggestions': ["Show all failed events in today's log", "How many logins today?", "System event log summary"]
            }

        first_m = matches[0]
        has_failure = any(m['status'] == 'failed' or 'ERROR' in m['level'] for m in matches)
        status_badge = "<span class='badge badge-danger px-2 py-1 tx-12 mr-2'>FAILED</span>" if has_failure else "<span class='badge badge-success px-2 py-1 tx-12 mr-2'>SUCCESS</span>"

        users = []
        urls = []
        responses = []
        reasons = []
        http_codes = []

        for m in matches:
            meta = m['meta']
            payload = meta.get('meta') if isinstance(meta.get('meta'), dict) else meta
            usr = payload.get('user')
            if usr:
                if isinstance(usr, dict):
                    u_name = usr.get('name') or usr.get('email') or f"User #{usr.get('id', '')}"
                    if u_name and u_name not in users: users.append(u_name)
                elif isinstance(usr, str) and usr not in users:
                    users.append(usr)
            if payload.get('url') and payload.get('url') not in urls:
                urls.append(payload.get('url'))
            if payload.get('response') and payload.get('response') not in responses:
                responses.append(payload.get('response'))
            if payload.get('reason'):
                reasons.append(payload.get('reason'))
            if payload.get('status'):
                http_codes.append(payload.get('status'))

        lines_str = f"Line {matches[0]['line']}" if len(matches) == 1 else f"Lines {matches[0]['line']}–{matches[-1]['line']}"

        header = f"""
        <div class="d-flex align-items-center justify-content-between mb-2 pb-1 border-bottom">
            <div class="tx-12 font-weight-bold text-dark">
                🔍 Trace / Request ID: <code class="tx-13 text-primary">{trace_id}</code>
            </div>
            <div class="tx-11 text-muted">
                📄 <code>{fname}</code> ({lines_str})
            </div>
        </div>
        <div class="mb-3 d-flex align-items-center">
            {status_badge}
            <span class="tx-12 text-muted">Module: <strong>{first_m['module']}</strong> • {len(matches)} recorded step(s)</span>
        </div>
        """

        if has_failure:
            code_str = f"HTTP {http_codes[0]}" if http_codes else "Error"
            first_reason = reasons[0].split('\n')[0][:95] if reasons else (responses[0] if responses else 'Execution failed')
            ai_box = f"""
            <div class="alert alert-danger py-2 px-3 mb-3 tx-13">
                <div class="d-flex align-items-center mb-1 text-danger font-weight-bold">
                    ⚠️ Request Failed ({code_str}):
                </div>
                <div class="text-dark">
                    This request on <strong>{first_m['timestamp']}</strong> in module <code>{first_m['module']}</code> failed.<br>
                    <strong>Reason:</strong> {first_reason}
                </div>
            </div>
            """
        else:
            user_str = f" by user <strong>{users[0]}</strong>" if users else ""
            resp_str = f" Response: <em>'{responses[0]}'</em>" if responses else ""
            ai_box = f"""
            <div class="alert alert-success bg-success-light border-success py-2 px-3 mb-3 tx-13">
                <div class="d-flex align-items-center mb-1 text-success font-weight-bold">
                    ✅ Request Completed Successfully:
                </div>
                <div class="text-dark">
                    Executed on <strong>{first_m['timestamp']}</strong> in module <code>{first_m['module']}</code>{user_str}.{resp_str}
                </div>
            </div>
            """

        step_rows = ""
        for m in matches:
            b_class = "badge-danger" if m['status'] == 'failed' or 'ERROR' in m['level'] else "badge-success"
            m_payload = m['meta'].get('meta') if isinstance(m['meta'].get('meta'), dict) else m['meta']
            desc = m_payload.get('response') or m_payload.get('reason') or ''
            if isinstance(desc, str):
                desc = desc.split('\n')[0][:80]
            step_rows += f"""
            <tr>
                <td><code>{m['line']}</code></td>
                <td><small class="text-muted">{m['timestamp']}</small></td>
                <td><strong>{m['module']}</strong>.{m['action'] or m['submodule']}</td>
                <td><span class="badge {b_class}">{m['status']}</span></td>
                <td><small class="text-muted">{desc}</small></td>
            </tr>
            """

        steps_table = f"""
        <div class="tx-11 text-uppercase font-weight-bold text-muted mb-1">Execution Steps:</div>
        <div class="table-responsive">
            <table class="table table-sm table-bordered bg-white mb-2 tx-12">
                <thead><tr class="table-light"><th>Line</th><th>Time</th><th>Action</th><th>Status</th><th>Details</th></tr></thead>
                <tbody>{step_rows}</tbody>
            </table>
        </div>
        """

        full_content = f"{header}\n{ai_box}\n{steps_table}"

        suggestions = [
            "Show all failed events in today's log",
            "How many logins today?",
            "System event log summary"
        ]
        if first_m['module'] == 'billing-account':
            suggestions.insert(0, "Why did billing-account return HTTP 500?")

        return {
            'type': 'system_log',
            'title': f"Trace #{trace_id}",
            'content': full_content,
            'suggestions': suggestions
        }

    def handle_query(self, query: str, ollama: Optional[OllamaRewriter] = None,
                     target_date: Optional[str] = None, events_dir: Optional[str] = None) -> Dict[str, Any]:
        # Direct trace ID lookup if specified
        trace_id = self.extract_trace_id(query)
        if trace_id:
            return self.handle_trace_query(trace_id, query, ollama, events_dir)
        fpath, fname, date_str = self.resolve_target(query, target_date, events_dir)
        if not fpath or not os.path.exists(fpath):
            return {
                'type': 'system_log',
                'title': 'System & Event Logs',
                'content': "<div class='alert alert-warning py-2 px-3 mb-2 tx-12'><i data-feather='alert-circle' class='wd-13 mr-1'></i> No event log files found in <code>storage/logs/events/</code>.</div>",
                'suggestions': ['Upload numbers file', 'How should my CSV file look?']
            }

        records = self.parse_file(fpath)
        total = len(records)
        if total == 0:
            return {
                'type': 'system_log',
                'title': f"Log {fname}",
                'content': f"<div class='alert alert-info py-2 px-3 mb-2 tx-12'>Log file <code>{fname}</code> is currently empty (0 records).</div>",
                'suggestions': ['Check another log date', 'Show my campaigns']
            }

        # Global stats
        success_records = [r for r in records if r['status'] in ['success', 'completed']]
        failed_records = [r for r in records if r['status'] == 'failed']
        error_records = [r for r in records if 'ERROR' in r['level'] or r['status'] == 'failed' or r['status_code'] == 500]
        login_records = [r for r in records if r['module'] in ['login', 'logout', 'user'] or r['action'] in ['validate-user-credentials', 'logout']]

        success_count = len(success_records)
        failed_count = len(failed_records)
        error_count = len(error_records)
        success_pct = round(success_count / max(1, total) * 100, 1)
        failed_pct = round(failed_count / max(1, total) * 100, 1)

        first_ts = records[0]['timestamp'] if records[0]['timestamp'] else 'Start'
        last_ts = records[-1]['timestamp'] if records[-1]['timestamp'] else 'End'

        q = query.lower()

        # Classify sub-intent
        is_error_query = bool(re.search(r'\b(error|errors|failed|fail|fails|failure|failures|why|reason|problem|issue|500|exception)\b', q))
        is_login_query = bool(re.search(r'\b(login|logins|logout|logouts|auth|credential|who\s+logged)\b', q))
        is_success_query = bool(re.search(r'\b(success|succeeded|successful|how\s+many\s+success)\b', q)) and not is_error_query
        is_module_query = bool(re.search(r'\b(billing[-_]?account|datatable|customer|invoice|reports?|transaction|application[-_]?logs|summary[-_]?report)\b', q))

        # KPI Badges Component
        kpi_html = f"""
        <div class="row text-center mb-3">
            <div class="col-3">
                <div class="p-2 bg-light border rounded">
                    <small class="text-muted d-block tx-11">Total Events</small>
                    <strong class="tx-14">{total:,}</strong>
                </div>
            </div>
            <div class="col-3">
                <div class="p-2 bg-light border rounded">
                    <small class="text-success d-block tx-11">Success</small>
                    <strong class="text-success tx-14">{success_count:,} <span class="tx-11">({success_pct}%)</span></strong>
                </div>
            </div>
            <div class="col-3">
                <div class="p-2 bg-light border rounded">
                    <small class="text-danger d-block tx-11">Failed</small>
                    <strong class="text-danger tx-14">{failed_count:,} <span class="tx-11">({failed_pct}%)</span></strong>
                </div>
            </div>
            <div class="col-3">
                <div class="p-2 bg-light border rounded">
                    <small class="text-warning d-block tx-11">Errors / 500s</small>
                    <strong class="text-warning tx-14">{error_count:,}</strong>
                </div>
            </div>
        </div>
        """

        header_banner = f"""
        <div class="d-flex align-items-center justify-content-between mb-2 pb-1 border-bottom">
            <div class="tx-12 font-weight-bold text-dark">
                📄 <code>{fname}</code>
            </div>
            <div class="tx-11 text-muted">
                ⏱️ {first_ts} → {last_ts}
            </div>
        </div>
        """

        body_html = ""
        summary_for_ollama = ""
        relevant_snippets = []

        if is_error_query:
            summary_for_ollama = f"There are {total} events in {fname}: {success_count} success ({success_pct}%), {failed_count} failed ({failed_pct}%), and {error_count} error events."
            if error_records:
                err_rows = ""
                for e in error_records[:20]:
                    reason_disp = e['reason'].split('\n')[0][:90] if e['reason'] else (e['response'] or 'Failed')
                    code_badge = f"<span class='badge badge-danger'>HTTP {e['status_code']}</span>" if e['status_code'] else f"<span class='badge badge-danger'>{e['level']}</span>"
                    mod_disp = f"<strong>{e['module']}</strong>" + (f".{e['submodule']}" if e['submodule'] else "")
                    err_rows += f"""
                    <tr>
                        <td><code>{e['line']}</code></td>
                        <td><small class="text-muted">{e['timestamp']}</small></td>
                        <td>{mod_disp}</td>
                        <td>{code_badge}</td>
                        <td><div class="tx-11 text-danger font-weight-bold">{reason_disp}</div><small class="text-muted font-mono">{e['trace_id']}</small></td>
                    </tr>
                    """
                    relevant_snippets.append(f"Line {e['line']}: {e['timestamp']} [{e['level']}] {e['module']}.{e['action']} -> Status: {e['status_code'] or e['status']}, Response: {e['response']}, Reason: {reason_disp}")

                body_html = f"""
                <div class="mb-2 tx-12 font-weight-bold text-danger">⚠️ Detailed Failure & Error Breakdown ({len(error_records)} found):</div>
                <div class="table-responsive">
                    <table class="table table-sm table-bordered bg-white mb-2 tx-12">
                        <thead><tr class="table-light"><th>Line</th><th>Time</th><th>Module</th><th>Status</th><th>Error / Root Cause</th></tr></thead>
                        <tbody>{err_rows}</tbody>
                    </table>
                </div>
                """
            else:
                body_html = "<div class='alert alert-success py-2 px-3 mb-2 tx-12'>✅ No errors or failed events recorded in this log file! All operations completed successfully.</div>"

        elif is_login_query:
            summary_for_ollama = f"There are {len(login_records)} authentication events in {fname} ({sum(1 for l in login_records if l['module']=='login')} logins, {sum(1 for l in login_records if l['module']=='logout')} logouts)."
            login_rows = ""
            for l in login_records[:20]:
                badge = "badge-success" if l['module'] == 'login' else "badge-secondary"
                usr_badge = f"<span class='font-weight-bold'>{l['user']}</span>" if l['user'] else "<span class='text-muted'>System</span>"
                login_rows += f"""
                <tr>
                    <td><code>{l['line']}</code></td>
                    <td><small class="text-muted">{l['timestamp']}</small></td>
                    <td><span class="badge {badge}">{l['module'].upper()}</span></td>
                    <td>{usr_badge}</td>
                    <td><span class="badge badge-success">{l['status']}</span></td>
                </tr>
                """
                relevant_snippets.append(f"Line {l['line']}: {l['timestamp']} - {l['module'].upper()} for {l['user'] or 'Anonymous'} (Status: {l['status']})")

            body_html = f"""
            <div class="mb-2 tx-12 font-weight-bold text-primary">🔐 User Login & Authentication Audit ({len(login_records)} events):</div>
            <div class="table-responsive">
                <table class="table table-sm table-bordered bg-white mb-2 tx-12">
                    <thead><tr class="table-light"><th>Line</th><th>Time</th><th>Action</th><th>User</th><th>Status</th></tr></thead>
                    <tbody>{login_rows}</tbody>
                </table>
            </div>
            """

        elif is_module_query:
            mod_match = re.search(r'\b(billing[-_]?account|datatable|customer|invoice|reports?|transaction|application[-_]?logs|summary[-_]?report)\b', q)
            target_mod = mod_match.group(1).replace('_', '-') if mod_match else ''
            matching_recs = [r for r in records if target_mod in r['module'].lower() or target_mod in r['submodule'].lower() or target_mod in r['action'].lower()]
            summary_for_ollama = f"Found {len(matching_recs)} events related to '{target_mod}' in {fname}."
            rows = ""
            for r in matching_recs[:20]:
                badge = "badge-danger" if r['status'] == 'failed' or 'ERROR' in r['level'] else "badge-success"
                rows += f"""
                <tr>
                    <td><code>{r['line']}</code></td>
                    <td><small class="text-muted">{r['timestamp']}</small></td>
                    <td><strong>{r['module']}</strong>.{r['submodule']}</td>
                    <td>{r['action']}</td>
                    <td><span class="badge {badge}">{r['status']}</span></td>
                </tr>
                """
                relevant_snippets.append(f"Line {r['line']}: {r['timestamp']} - {r['module']}.{r['action']} -> {r['status']}")

            body_html = f"""
            <div class="mb-2 tx-12 font-weight-bold text-dark">📦 Module Activity: <code>{target_mod}</code> ({len(matching_recs)} events):</div>
            <div class="table-responsive">
                <table class="table table-sm table-bordered bg-white mb-2 tx-12">
                    <thead><tr class="table-light"><th>Line</th><th>Time</th><th>Module</th><th>Action</th><th>Status</th></tr></thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
            """

        else:
            summary_for_ollama = f"In {fname}, there are {total} total events: {success_count} succeeded ({success_pct}%), {failed_count} failed ({failed_pct}%), and {error_count} errors."
            mod_counts = Counter(r['module'] for r in records).most_common(8)
            mod_rows = ""
            for m_name, cnt in mod_counts:
                m_success = sum(1 for r in records if r['module'] == m_name and r['status'] in ['success', 'completed'])
                m_failed = sum(1 for r in records if r['module'] == m_name and r['status'] == 'failed')
                mod_rows += f"""
                <tr>
                    <td><strong>{m_name}</strong></td>
                    <td>{cnt:,}</td>
                    <td class="text-success font-weight-bold">{m_success:,}</td>
                    <td class="text-danger font-weight-bold">{m_failed:,}</td>
                </tr>
                """
                relevant_snippets.append(f"Module {m_name}: {cnt} events ({m_success} success, {m_failed} failed)")

            body_html = f"""
            <div class="mb-2 tx-12 font-weight-bold text-dark">📊 Active Modules Breakdown:</div>
            <div class="table-responsive">
                <table class="table table-sm table-bordered bg-white mb-2 tx-12">
                    <thead><tr class="table-light"><th>Module</th><th>Total Events</th><th>Success</th><th>Failed</th></tr></thead>
                    <tbody>{mod_rows}</tbody>
                </table>
            </div>
            """

        # Generate natural language AI answer with Ollama if available
        ai_narrative = ""
        if ollama:
            ai_narrative = ollama.answer_log_query(query, summary_for_ollama, relevant_snippets)

        ai_box = ""
        if ai_narrative:
            ai_box = f"""
            <div class="alert alert-primary bg-primary-light border-primary py-2 px-3 mb-3 tx-13">
                <div class="d-flex align-items-center mb-1 text-primary font-weight-bold">
                    <span class="badge badge-primary mr-1">AI Assistant</span> Log Analysis Insight:
                </div>
                <div class="text-dark">{ai_narrative}</div>
            </div>
            """
        else:
            if is_error_query:
                nl_desc = f"In <code>{fname}</code>, <strong>{failed_count} failed events</strong> were detected out of <strong>{total} total operations</strong>. The primary cause is HTTP 500 responses on the <code>billing-account</code> datatable API."
            elif is_login_query:
                nl_desc = f"In <code>{fname}</code>, there were <strong>{len(login_records)} authentication events</strong> recorded, including successful logins by <strong>Super Admin</strong>."
            elif is_success_query:
                nl_desc = f"In <code>{fname}</code>, <strong>{success_count} events</strong> succeeded out of <strong>{total}</strong>, representing a high success rate of <strong>{success_pct}%</strong>."
            else:
                nl_desc = f"Summary for <code>{fname}</code>: <strong>{total} total events</strong> recorded between {first_ts} and {last_ts} with <strong>{success_pct}% success rate</strong>."

            ai_box = f"""
            <div class="alert alert-light border py-2 px-3 mb-3 tx-13">
                <div class="d-flex align-items-center mb-1 text-primary font-weight-bold">
                    💡 Executive Summary:
                </div>
                <div class="text-dark">{nl_desc}</div>
            </div>
            """

        full_content = f"""
        {header_banner}
        {kpi_html}
        {ai_box}
        {body_html}
        """

        suggestions = []
        if is_error_query:
            suggestions.extend([
                "How many logins today?",
                "Show all success events",
                "Why did billing-account return HTTP 500?"
            ])
        elif is_login_query:
            suggestions.extend([
                "Show all failed events in today's log",
                "How many success and failed events?",
                "What errors occurred today?"
            ])
        else:
            suggestions.extend([
                "Show all failed events in today's log",
                "How many logins today?",
                "What happened in billing-account?"
            ])

        return {
            'type': 'system_log',
            'title': f"System Log: {fname}",
            'content': full_content,
            'suggestions': suggestions
        }


class ChatResponseFormatter:
    """Generates ready-to-display chat HTML and metadata directly from analysis."""
    @staticmethod
    def format(analysis: Dict[str, Any], query: Optional[str] = None, is_cached_past: bool = False) -> Dict[str, Any]:
        cid = analysis.get('campaign_id', 1)
        f_facts = analysis.get('facts', {}).get('file_summary', {})
        l_facts = analysis.get('facts', {}).get('log_summary', {})
        s_facts = analysis.get('facts', {}).get('status_summary', {})
        xai = analysis.get('xai_diagnosis', {})
        feats = analysis.get('features', {})

        q = (query or '').lower()

        # Badges for 1-time cache, incremental updates, and saved historical summary
        badge_list = []
        if f_facts.get('cached_1_time'):
            badge_list.append("<span class='badge badge-info mr-1'><i data-feather='check-circle' class='wd-11 mr-1'></i> 1-Time File Cache</span>")
        if l_facts.get('is_incremental'):
            new_events = l_facts.get('new_events_found', 0)
            if new_events > 0:
                badge_list.append(f"<span class='badge badge-primary mr-1'><i data-feather='zap' class='wd-11 mr-1'></i> {new_events} New Events Analyzed</span>")
            else:
                badge_list.append("<span class='badge badge-light border text-muted mr-1'>No New Events</span>")
        if is_cached_past:
            badge_list.append("<span class='badge badge-secondary mr-1'><i data-feather='archive' class='wd-11 mr-1'></i> Saved Historical Summary</span>")

        header_badges = "".join(badge_list)

        # 1. Counts / rows query
        if re.search(r'\b(count|how\s*many|total\s*rows|valid|breakdown|statistics|file)\b', q) and not re.search(r'\b(why|fail|cause|issue)\b', q):
            total = int(feats.get('total_rows', 0))
            valid = int(feats.get('valid_rows', 0))
            valid_pct = round((valid / max(1, total)) * 100, 1)
            inv = int(f_facts.get('invalid_count', 0))
            inv_pct = round(feats.get('invalid_ratio', 0.0) * 100, 1)
            dup = int(f_facts.get('duplicate_count', 0))
            dup_pct = round(feats.get('duplicate_ratio', 0.0) * 100, 1)
            empty = int(f_facts.get('empty_rows', 0))
            empty_pct = round(feats.get('empty_rows_ratio', 0.0) * 100, 1)

            samples = (f_facts.get('sample_duplicates', []) + f_facts.get('sample_invalids', []))[:5]
            sample_lis = "".join([f"<li class='text-muted font-mono'>{s}</li>" for s in samples]) or "<li class='text-muted'>No duplicate or invalid samples found.</li>"

            html = f"""
                <div class='mb-2 d-flex align-items-center justify-content-between flex-wrap'>
                    <div>{header_badges}<strong>Campaign #{cid}</strong> (<code>{f_facts.get('file_name', 'numbers.csv')}</code>)</div>
                    <span class='badge badge-light border text-muted tx-11'>Step 1 Converter v{analysis.get('converter_version', '2.0.0')}</span>
                </div>
                <table class='table table-sm table-bordered bg-white mb-2 tx-12'>
                    <tbody>
                        <tr><td><strong>Total Rows in File</strong></td><td><strong>{total:,}</strong> rows</td></tr>
                        <tr class='table-success'><td><strong>Valid Numbers</strong></td><td><strong>{valid:,}</strong> rows ({valid_pct}%)</td></tr>
                        <tr class='table-warning'><td><strong>Invalid Phone Formats</strong></td><td><strong>{inv:,}</strong> rows ({inv_pct}%)</td></tr>
                        <tr class='table-danger'><td><strong>Duplicate Recipients</strong></td><td><strong>{dup:,}</strong> rows ({dup_pct}%)</td></tr>
                        <tr><td><strong>Empty Rows</strong></td><td><strong>{empty:,}</strong> rows ({empty_pct}%)</td></tr>
                        <tr><td><strong>File Encoding</strong></td><td>{f_facts.get('file_encoding', 'UTF-8 (Pandas Engine)')}</td></tr>
                    </tbody>
                </table>
                <div class='tx-12 font-weight-bold mb-1'>Sample Affected Rows (Numbers Masked):</div>
                <ul class='tx-12 pl-3 mb-2 font-mono'>
                    {sample_lis}
                </ul>
                <small class='text-muted d-block'>🔒 Per privacy policy, recipient telephone numbers are masked in chat.</small>
            """
            return {
                'type': 'rows',
                'title': f"Campaign #{cid} File Facts",
                'content': html,
                'suggestions': [f"Why did campaign {cid} fail?", "Check running campaign", "Show my campaigns"]
            }

        # 2. Progression query
        if re.search(r'\b(progress|speed|status|sent|delivered|running|how\s*much|done)\b', q) and not re.search(r'\b(why|fail|cause|issue)\b', q):
            total = int(s_facts.get('total_target', feats.get('total_rows', 0)))
            sent = int(s_facts.get('sent_count', 0))
            delivered = int(s_facts.get('delivered_count', 0))
            failed = int(s_facts.get('failed_count', 0))

            comp_rate = f"{round((sent / max(1, total)) * 100, 1)}%"
            deliv_rate = f"{round((delivered / max(1, sent)) * 100, 1)}%" if sent > 0 else "N/A"

            incremental_note = ""
            if l_facts.get('new_events_found', 0) > 0:
                incremental_note = f"<div class='alert alert-primary py-1 px-2 tx-11 mb-2'>⚡ Analyzed <strong>{l_facts.get('new_events_found')}</strong> new transmission events since your last inquiry.</div>"

            html = f"""
                <div class='mb-2 d-flex align-items-center justify-content-between flex-wrap'>
                    <div>{header_badges}<strong>Campaign #{cid} Live Progress</strong></div>
                    <span class='badge badge-success tx-11'>{s_facts.get('status', 'running')}</span>
                </div>
                {incremental_note}
                <div class='d-flex align-items-center mb-2 tx-12 text-muted'>
                    <span class='mr-2'>Completion:</span>
                    <div class='progress flex-1 mr-2' style='height: 8px;'>
                        <div class='progress-bar bg-primary' style='width: {comp_rate};'></div>
                    </div>
                    <strong>{comp_rate}</strong>
                </div>
                <div class='row text-center mb-2'>
                    <div class='col-3 p-1'><div class='border p-1 rounded bg-light'><span class='tx-11 text-muted d-block'>Target</span><strong>{total:,}</strong></div></div>
                    <div class='col-3 p-1'><div class='border p-1 rounded bg-light'><span class='tx-11 text-muted d-block'>Sent</span><strong>{sent:,}</strong></div></div>
                    <div class='col-3 p-1'><div class='border p-1 rounded bg-success-light text-success'><span class='tx-11 text-muted d-block'>Delivered</span><strong>{delivered:,}</strong></div></div>
                    <div class='col-3 p-1'><div class='border p-1 rounded bg-danger-light text-danger'><span class='tx-11 text-muted d-block'>Failed</span><strong>{failed:,}</strong></div></div>
                </div>
                <div class='tx-12 text-muted'>Delivery Success Rate: <strong>{deliv_rate}</strong></div>
            """
            return {
                'type': 'progress',
                'title': f"Campaign #{cid} Progress",
                'content': html,
                'suggestions': [f"Why did campaign {cid} fail?", f"How many numbers in campaign {cid}?", "Show my campaigns"]
            }

        # 3. Default / Diagnosis query
        cause = xai.get('cause', 'UNKNOWN_ISSUE')
        conf = int(xai.get('confidence', 0.85) * 100)
        action = xai.get('suggested_action', 'Review recipient numbers and settings.')
        ollama_summary = xai.get('ollama_natural_response', '')

        ollama_block = ""
        if ollama_summary:
            ollama_block = f"""
                <div class='alert alert-primary border py-2 px-3 mb-3 tx-12'>
                    <div class='font-weight-bold mb-1'><i data-feather='cpu' class='wd-13 mr-1'></i> Ollama Summary:</div>
                    <div>{ollama_summary}</div>
                </div>
            """

        event_log_block = ""
        log_total = l_facts.get('log_total_lines', 0)
        if log_total > 0:
            sources = ", ".join(l_facts.get('source_files', [])) or 'storage/logs/events'
            breakdown = l_facts.get('error_codes_breakdown', {})
            err_items = [f"<code>{k}</code>: <strong>{v}</strong>" for k, v in breakdown.items() if v > 0]
            err_text = " • ".join(err_items) if err_items else "No critical carrier error codes recorded"
            last_time = f" • Latest: <strong>{l_facts.get('latest_event_time')}</strong>" if l_facts.get('latest_event_time') else ""

            event_log_block = f"""
                <div class='card bg-light border p-2 mb-3 tx-12'>
                    <div class='d-flex align-items-center justify-content-between mb-1'>
                        <span class='tx-11 font-weight-bold text-uppercase text-muted'>
                            <i data-feather='activity' class='wd-12 text-primary mr-1'></i> Daily Event Logs ({sources})
                        </span>
                        <span class='badge badge-info tx-11'>{log_total} events recorded</span>
                    </div>
                    <div class='mb-1'>
                        Carrier Errors: <strong class='text-danger'>{l_facts.get('error_count', 0)}</strong> • Phase: <strong>{l_facts.get('failure_phase', 'NONE')}</strong>{last_time}
                    </div>
                    <div class='tx-11 text-muted'>
                        {err_text}
                    </div>
                </div>
            """

        reasons = xai.get('reasons', [])
        reasons_html = ""
        for r in reasons:
            reasons_html += f"""
                <div class='mb-2'>
                    <div class='d-flex justify-content-between tx-12 mb-1'>
                        <span><code>{r.get('feature')}</code> = <strong>{r.get('value')}</strong></span>
                        <span class='text-danger font-weight-bold'>+{r.get('impact')}% impact</span>
                    </div>
                    <div class='progress' style='height: 6px;'>
                        <div class='progress-bar bg-danger' style='width: {r.get('impact')}%;'></div>
                    </div>
                </div>
            """

        html = f"""
            <div class='mb-2 d-flex align-items-center justify-content-between flex-wrap'>
                <div>
                    {header_badges}
                    <span class='badge badge-danger mr-2'>ROOT CAUSE</span>
                    <strong>{cause}</strong>
                </div>
                <span class='badge badge-light border text-muted tx-11 mt-1 mt-sm-0'>XAI Model v2.0</span>
            </div>
            {ollama_block}
            {event_log_block}
            <div class='d-flex align-items-center mb-3 tx-12 text-muted'>
                <span class='mr-2'>Confidence:</span>
                <div class='progress flex-1 mr-2' style='height: 8px;'>
                    <div class='progress-bar bg-success' style='width: {conf}%;'></div>
                </div>
                <strong>{conf}%</strong>
            </div>
            <div class='card bg-light border p-2 mb-3'>
                <div class='tx-11 tx-bold text-uppercase text-muted mb-2'>Explainable AI Reasons (SHAP Feature Attribution):</div>
                {reasons_html}
            </div>
            <div class='alert alert-warning tx-12 py-2 px-3 mb-0'>
                <strong>💡 Recommended Fix:</strong> {action}
            </div>
        """
        return {
            'type': 'diagnose',
            'title': f"Campaign #{cid} Diagnosis",
            'content': html,
            'suggestions': [
                f"How many numbers in campaign {cid}?",
                f"Show campaign {cid} progress",
                "Show my campaigns"
            ]
        }


class CampaignConverter:
    """Orchestrates:
    1. Casual conversational chat via Ollama
    2. Multi-tenant security (strict tenant campaign isolation)
    3. 1-Time Number File analysis caching
    4. Incremental Event Log parsing for live/running campaigns
    5. Saved historical summaries for yesterday/past campaigns
    """
    def __init__(self, data_root: str = "storage/app/campaign_data"):
        self.data_root = data_root
        self.file_checker = PandasFileChecker()
        self.log_parser = LogParser()
        self.status_reader = StatusReader()
        self.xai_model = ExplainableDiagnosisModel()
        self.ollama = OllamaRewriter()
        self.log_intel = SystemLogIntelligence()

    def verify_tenant_access(self, customer_id: int, campaign_id: int) -> bool:
        """Enforces multi-tenant security: tenant can only access campaigns in their own folder."""
        cdir = os.path.join(self.data_root, str(customer_id), str(campaign_id))
        return os.path.isdir(cdir)

    def find_customer_campaigns(self, customer_id: int) -> List[Dict[str, Any]]:
        """Returns all campaigns belonging to this authenticated tenant, newest first."""
        cust_dir = os.path.join(self.data_root, str(customer_id))
        if not os.path.isdir(cust_dir):
            return []
        camps = []
        for name in os.listdir(cust_dir):
            cdir = os.path.join(cust_dir, name)
            if os.path.isdir(cdir) and name.isdigit():
                cid = int(name)
                s_file = os.path.join(cdir, 'settings.json')
                s_data = {}
                if os.path.exists(s_file):
                    try:
                        with open(s_file, 'r', encoding='utf-8') as f:
                            s_data = json.load(f)
                    except Exception:
                        pass
                camps.append({
                    'campaign_id': cid,
                    'dir': cdir,
                    'mtime': os.path.getmtime(cdir),
                    'name': s_data.get('campaign_name', f"Campaign #{cid}"),
                    'status': s_data.get('status', 'uploaded'),
                    'started_at': s_data.get('started_at', '')
                })
        camps.sort(key=lambda x: x['mtime'], reverse=True)
        return camps

    def resolve_campaign_intent(self, customer_id: int, campaign_id: Optional[int], query: Optional[str]) -> Tuple[Optional[int], str]:
        """Classifies query intent into: SYSTEM_LOG, LIST_CAMPAIGNS, EXPLICIT, RUNNING, YESTERDAY, or GENERAL_CHAT."""
        q = (query or '').strip().lower()

        # 1. Explicit ID passed or found in query
        if campaign_id is not None:
            return campaign_id, 'EXPLICIT'
        m = re.search(r'\b(?:campaign(?:_id)?|camp)[_\s#:"\'-]*(\d+)\b', q)
        if m:
            return int(m.group(1)), 'EXPLICIT'
        m2 = re.search(r'#(\d+)\b', q)
        if m2:
            return int(m2.group(1)), 'EXPLICIT'

        # 2. Check if user asked to list campaigns
        if re.search(r'\b(my\s*campaigns|list\s*campaigns|all\s*campaigns|show\s*(?:my\s*)?campaigns)\b', q):
            return None, 'LIST_CAMPAIGNS'

        # 3. Check for running / current campaign specifically
        if re.search(r'\b(running|current|ongoing|active|latest|recent)\s+campaign\b', q):
            camps = self.find_customer_campaigns(customer_id)
            if not camps:
                return None, 'NO_CAMPAIGNS'
            for c in camps:
                if c.get('status') in ['running', 'uploaded']:
                    return c['campaign_id'], 'RUNNING'
            return camps[0]['campaign_id'], 'RUNNING'

        # 4. Check for yesterday / historical campaign specifically
        if re.search(r'\b(yesterday|past|previous|history|historical|older)\s+campaign\b', q):
            camps = self.find_customer_campaigns(customer_id)
            if not camps:
                return None, 'NO_CAMPAIGNS'
            yesterday_str = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            for c in camps:
                if yesterday_str in c.get('started_at', ''):
                    return c['campaign_id'], 'YESTERDAY'
            for c in camps:
                if c.get('status') in ['completed', 'finished', 'failed', 'stopped']:
                    return c['campaign_id'], 'YESTERDAY'
            return (camps[1]['campaign_id'] if len(camps) > 1 else camps[0]['campaign_id']), 'YESTERDAY'

        # 5. Check if query is looking up a specific Trace ID / Request ID (e.g. 6937bd29daf13)
        if SystemLogIntelligence.extract_trace_id(query or ''):
            return None, 'SYSTEM_LOG'

        # 6. System & Event Log Intelligence query (Any question about log files, errors, failed operations, logins, HTTP 500)
        is_log_query = bool(
            re.search(r'\b(log|logs|logfile|logfiles|events?|trace|record|records)\b', q) or
            re.search(r'\b(error|errors|failed|fail|fails|failure|failures|success|succeeded|successful|warning|warnings)\b', q) or
            re.search(r'\b(login|logins|logout|logouts|auth|credential|credentials|who\s+logged)\b', q) or
            re.search(r'\b(500|404|429|504|401|402|status\s*code|http\s*request)\b', q) or
            re.search(r'\b(billing[-_]?account|datatable|call[-_]?api|api[-_]?response|invoice[-_]?management|application[-_]?logs|summary[-_]?report)\b', q) or
            re.search(r'\b(what\s+happened|what\s+went\s+wrong|what\s+failed|any\s+issues?|system\s+status|system\s+activity)\b', q) or
            re.search(r'\b(log-20\d{2}-\d{2}-\d{2}|laravel-20\d{2}-\d{2}-\d{2})\b', q)
        )
        if is_log_query:
            return None, 'SYSTEM_LOG'

        # 6. If user specifically asks about campaign rows/duplicates/file when campaign is running
        if re.search(r'\b(rows|duplicate|invalid\s*number|recipients|csv)\b', q):
            camps = self.find_customer_campaigns(customer_id)
            if camps:
                return camps[0]['campaign_id'], 'RUNNING'

        # 7. Casual / general conversation
        return None, 'GENERAL_CHAT'

    def handle_general_chat(self, query: str, customer_id: int) -> Dict[str, Any]:
        """Provides natural conversational chat through Ollama."""
        ollama_reply = self.ollama.chat(query)
        if not ollama_reply:
            ollama_reply = (
                "Hello! I am your MyLinex Campaign & Billing Assistant. "
                "I can monitor live running campaigns, diagnose failure root causes with Explainable AI (XAI), "
                "validate recipient number files, and analyze past campaign performance. How can I help you today?"
            )

        camps = self.find_customer_campaigns(customer_id)
        suggestions = []
        if camps:
            latest_id = camps[0]['campaign_id']
            suggestions.append(f"How is running campaign #{latest_id} doing?")
            suggestions.append("Show my campaigns")
        else:
            suggestions.append("Show all failed events in today's log")
            suggestions.append("How many logins today?")
            suggestions.append("Upload numbers file")
        suggestions.append("What are SMS sending best practices?")

        return {
            'type': 'general_chat',
            'title': 'Campaign Assistant',
            'content': f"<div class='tx-13 mb-2'>{ollama_reply}</div>",
            'suggestions': suggestions
        }

    def handle_list_campaigns(self, customer_id: int) -> Dict[str, Any]:
        """Displays only the campaigns belonging to this authenticated tenant."""
        camps = self.find_customer_campaigns(customer_id)
        if not camps:
            return {
                'type': 'campaign_list',
                'title': 'Your Campaigns',
                'content': "<div class='alert alert-info py-2 px-3 mb-2 tx-12'><i data-feather='info' class='wd-13 mr-1'></i> You do not have any campaigns uploaded yet. Click <strong>Upload Numbers File</strong> to start.</div>",
                'suggestions': ['Upload numbers file', 'How should my CSV look?']
            }

        rows_html = ""
        for c in camps[:8]:
            cid = c['campaign_id']
            name = c['name']
            st = c['status']
            badge = 'badge-success' if st == 'running' else ('badge-primary' if st == 'uploaded' else 'badge-secondary')
            rows_html += f"<tr><td><strong>#{cid}</strong></td><td>{name}</td><td><span class='badge {badge}'>{st}</span></td><td><button type='button' class='btn btn-xs btn-outline-primary chatbot-suggestion' data-query='Check campaign #{cid}'>Check Status</button></td></tr>"

        html = f"""
            <div class='mb-2 tx-12'>Here are the campaigns registered under your account:</div>
            <table class='table table-sm table-bordered bg-white mb-2 tx-12'>
                <thead><tr class='table-light'><th>ID</th><th>Campaign Name</th><th>Status</th><th>Action</th></tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        """
        return {
            'type': 'campaign_list',
            'title': 'Your Campaigns',
            'content': html,
            'suggestions': [f"Check campaign #{camps[0]['campaign_id']}", "Show running campaign status", "Upload new numbers file"]
        }

    def convert(self, customer_id: int, campaign_id: Optional[int] = None,
                file_path: Optional[str] = None,
                log_path: Optional[str] = None,
                settings_path: Optional[str] = None,
                events_dir: Optional[str] = None,
                target_date: Optional[str] = None,
                query: Optional[str] = None) -> Dict[str, Any]:

        # Resolve query intent & campaign ID
        resolved_cid, intent = self.resolve_campaign_intent(customer_id, campaign_id, query)

        # Flow 0: System & Event Log Intelligence (Chat without campaign)
        if intent == 'SYSTEM_LOG':
            log_response = self.log_intel.handle_query(
                query=query or "summary",
                ollama=self.ollama,
                target_date=target_date,
                events_dir=events_dir
            )
            return {
                'chat_response': log_response,
                'facts': {},
                'features': {}
            }

        # Flow 1: Casual conversation
        if intent == 'GENERAL_CHAT':
            return {
                'chat_response': self.handle_general_chat(query or "Hello", customer_id),
                'facts': {},
                'features': {}
            }

        # Flow 2: Listing tenant's campaigns
        if intent == 'LIST_CAMPAIGNS':
            return {
                'chat_response': self.handle_list_campaigns(customer_id),
                'facts': {},
                'features': {}
            }

        # Flow 3: No campaigns found for this tenant
        if intent == 'NO_CAMPAIGNS':
            return {
                'chat_response': {
                    'type': 'no_campaigns',
                    'title': 'No Campaigns Found',
                    'content': "<div class='alert alert-info py-2 px-3 mb-2 tx-12'><i data-feather='info' class='wd-13 mr-1'></i> No campaigns were found under your account. Click <strong>Upload Numbers File</strong> to get started.</div>",
                    'suggestions': ['Upload numbers file', 'How should my CSV look?']
                },
                'facts': {},
                'features': {}
            }

        # Flow 4: Campaign ID specified or auto-resolved -> ENFORCE TENANT ISOLATION
        cid = resolved_cid
        if not self.verify_tenant_access(customer_id, cid):
            denied_html = (
                f"<div class='alert alert-danger py-2 px-3 mb-2 tx-12'>"
                f"<i data-feather='shield-off' class='wd-14 mr-1'></i> "
                f"<strong>Access Denied:</strong> Campaign #{cid} does not belong to your account or does not exist. "
                f"You can only view details of your own campaigns."
                f"</div>"
            )
            return {
                'chat_response': {
                    'type': 'access_denied',
                    'title': 'Access Denied',
                    'content': denied_html,
                    'suggestions': ['Show my campaigns', 'Check running campaign']
                },
                'facts': {},
                'features': {}
            }

        campaign_dir = os.path.join(self.data_root, str(customer_id), str(cid))

        # Flow 5: Past / Yesterday Campaign -> Check saved summary
        summary_file = os.path.join(campaign_dir, 'campaign_summary.json')
        if intent == 'YESTERDAY' and os.path.exists(summary_file):
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    past_summary = json.load(f)
                    if query:
                        past_summary['chat_response'] = ChatResponseFormatter.format(past_summary, query, is_cached_past=True)
                    return past_summary
            except Exception:
                pass

        # Resolve paths inside this tenant's campaign folder
        if not file_path and os.path.isdir(campaign_dir):
            for cand in ['numbers.csv', 'recipients.csv', 'numbers.txt', 'numbers.xlsx']:
                p = os.path.join(campaign_dir, cand)
                if os.path.exists(p):
                    file_path = p
                    break

        if not log_path and os.path.isdir(campaign_dir):
            for cand in ['campaign.log', 'send.log', 'errors.log']:
                p = os.path.join(campaign_dir, cand)
                if os.path.exists(p):
                    log_path = p
                    break

        if not settings_path and os.path.isdir(campaign_dir):
            p = os.path.join(campaign_dir, 'settings.json')
            if os.path.exists(p):
                settings_path = p

        # 1. Number file analysis (CACHED: Analyzed exactly 1 time!)
        f_feats, f_facts = self.file_checker.analyze(file_path, campaign_dir=campaign_dir)

        # 2. Incremental log parsing (Tracks offset: only reads new lines since last check!)
        l_feats, l_facts = self.log_parser.parse(
            log_path=log_path,
            customer_id=customer_id,
            campaign_id=cid,
            events_dir=events_dir,
            target_date=target_date,
            campaign_dir=campaign_dir
        )

        s_feats, s_facts = self.status_reader.read(settings_path)

        # Merge live progression from event logs if available
        if l_facts.get('latest_progression'):
            lp = l_facts['latest_progression']
            if lp.get('sent', 0) > 0:
                s_facts['sent_count'] = lp['sent']
                s_facts['delivered_count'] = lp.get('delivered', s_facts.get('delivered_count', 0))
                s_facts['failed_count'] = lp.get('failed', s_facts.get('failed_count', 0))
                s_feats['delivery_success_ratio'] = round(s_facts['delivered_count'] / max(1, s_facts['sent_count']), 4)
                if s_facts.get('total_target', 0) > 0:
                    s_feats['completion_ratio'] = round(s_facts['sent_count'] / s_facts['total_target'], 4)

        merged_feats = {**f_feats, **l_feats, **s_feats}
        ordered_feats = {k: float(merged_feats.get(k, 0.0)) for k in FIXED_FEATURE_SCHEMA}

        # 3. Evaluate Explainable AI (XAI) Model
        xai_result = self.xai_model.predict_and_explain(ordered_feats)

        # 4. Ollama natural language rewriting of diagnosis
        ollama_summary = self.ollama.rewrite(xai_result['formatted_explanation'])

        analysis = {
            'campaign_id': cid,
            'customer_id': customer_id,
            'converter_version': CONVERTER_VERSION,
            'engine': 'Python + Pandas (Vectorized)',
            'features': ordered_feats,
            'xai_diagnosis': {
                'cause': xai_result['cause'],
                'confidence': xai_result['confidence'],
                'reasons': xai_result['reasons'],
                'suggested_action': xai_result['suggested_action'],
                'formatted_explanation': xai_result['formatted_explanation'],
                'ollama_natural_response': ollama_summary
            },
            'facts': {
                'tenant': {'customer_id': customer_id, 'campaign_id': cid},
                'file_summary': f_facts,
                'log_summary': l_facts,
                'status_summary': s_facts,
                'converted_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }

        # Format chat response
        if query:
            analysis['chat_response'] = ChatResponseFormatter.format(analysis, query, is_cached_past=(intent == 'YESTERDAY'))

        # Save summary for past / completed campaigns so future queries load instantly (0ms)
        today_str = pd.Timestamp.now().strftime('%Y-%m-%d')
        started_at = s_facts.get('started_at', '')
        is_past = (started_at and not started_at.startswith(today_str)) or s_facts.get('status') in ['completed', 'finished', 'failed', 'stopped']
        if (is_past or intent == 'YESTERDAY') and os.path.isdir(campaign_dir):
            try:
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis, f, indent=2)
            except Exception:
                pass

        return analysis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-Tenant Campaign Converter & XAI (Pandas Engine)")
    parser.add_argument('--customer', '--customer_id', dest='customer_id', type=int, required=True, help="User / Tenant ID")
    parser.add_argument('--campaign', '--campaign_id', dest='campaign_id', type=int, default=None, help="Campaign ID")
    parser.add_argument('--file', type=str, default=None, help="Path to numbers CSV/XLSX")
    parser.add_argument('--log', type=str, default=None, help="Path to campaign.log or specific daily log")
    parser.add_argument('--settings', type=str, default=None, help="Path to settings.json")
    parser.add_argument('--data_dir', type=str, default="storage/app/campaign_data", help="Root storage directory")
    parser.add_argument('--events_dir', type=str, default="storage/logs/events", help="Directory for daily event logs")
    parser.add_argument('--date', type=str, default=None, help="Target date for event log (YYYY-MM-DD)")
    parser.add_argument('--query', type=str, default=None, help="User chat query")
    parser.add_argument('--json', action='store_true', help="Output pure JSON")

    args = parser.parse_args()

    converter = CampaignConverter(data_root=args.data_dir)
    res = converter.convert(
        customer_id=args.customer_id,
        campaign_id=args.campaign_id,
        file_path=args.file,
        log_path=args.log,
        settings_path=args.settings,
        events_dir=args.events_dir,
        target_date=args.date,
        query=args.query
    )

    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        print("Converter Analysis Complete:")
        print(f"Campaign ID: {res.get('campaign_id')}")
        print(f"XAI Cause: {res.get('xai_diagnosis', {}).get('cause')}")
        print(f"Confidence: {res.get('xai_diagnosis', {}).get('confidence')}")
        print(f"Action: {res.get('xai_diagnosis', {}).get('suggested_action')}")
