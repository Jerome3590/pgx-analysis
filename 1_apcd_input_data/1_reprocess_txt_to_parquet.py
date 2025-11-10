import argparse
import os
import sys
import time
import logging
from typing import List, Tuple, Optional
import boto3
import codecs
import json

# Add project root to path (helpers folder is at pgx-analysis level)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from helpers_1997_13.logging_utils import setup_logging, save_logs_to_s3
    from helpers_1997_13.duckdb_utils import create_simple_duckdb_connection
    from helpers_1997_13.pipeline_state import PipelineState
except Exception:
    setup_logging = None
    save_logs_to_s3 = None
    create_simple_duckdb_connection = None
    PipelineState = None




def setup_logger() -> logging.Logger:
    logger = logging.getLogger("reprocess_txt_to_parquet")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    if not logger.handlers:
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def s3_exists(bucket: str, key: str) -> bool:
    s3 = boto3.client("s3")
    try:
        resp = s3.head_object(Bucket=bucket, Key=key)
        return resp.get("ContentLength", 0) > 0
    except s3.exceptions.ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "403", "NoSuchKey"):
            return False
        raise


def _parse_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    rest = uri[5:]
    parts = rest.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def list_reject_files(bronze_root: str, dataset: str) -> List[str]:
    s3 = boto3.client("s3")
    bucket, root_key = _parse_s3_uri(bronze_root)
    prefix = f"{root_key.rstrip('/')}/_rejects/{dataset}/" if root_key else f"_rejects/{dataset}/"
    rejects: List[str] = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".rejects.txt"):
                rejects.append(f"s3://{bucket}/{key}")
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return rejects


def get_etag(bucket: str, key: str) -> str:
    s3 = boto3.client("s3")
    head = s3.head_object(Bucket=bucket, Key=key)
    etag = head.get("ETag", "").strip('"')
    return etag


def convert_rejects_to_parquet(input_uri: str, output_uri: str, duckdb_threads: int, tmp_dir: Optional[str], proceed_on_errors: bool, logger: logging.Logger) -> None:
    # Create DuckDB connection consistent with pipeline
    if create_simple_duckdb_connection is None:
        import duckdb
        con = duckdb.connect()
    else:
        con = create_simple_duckdb_connection(logger, tmp_dir)

    try:
        # Threads per worker
        try:
            con.sql(f"PRAGMA threads={int(duckdb_threads)}")
        except Exception:
            con.sql("PRAGMA threads=1")

        # Inject AWS creds explicitly from boto3 (robust in EC2 workers)
        try:
            import boto3 as _b3
            _sess = _b3.Session()
            _creds = _sess.get_credentials()
            _region = _sess.region_name or "us-east-1"
            if _creds is not None:
                _f = _creds.get_frozen_credentials()
                if getattr(_f, "access_key", None) and getattr(_f, "secret_key", None):
                    con.sql(f"SET s3_access_key_id='{_f.access_key}'")
                    con.sql(f"SET s3_secret_access_key='{_f.secret_key}'")
                    if getattr(_f, "token", None):
                        con.sql(f"SET s3_session_token='{_f.token}'")
            con.sql(f"SET s3_region='{_region}'")
            con.sql("SET s3_url_style='path'")
        except Exception as _inj_e:
            logger.warning(f"Could not inject AWS credentials into DuckDB: {_inj_e}")

        # Build COPY using read_csv with forced options for speed; fallback to auto
        def make_copy_sql_forced(delim: str = '|', enc: str = 'CP1252') -> str:
            opts = [
                f"DELIM='{delim}'",
                "HEADER=TRUE",
                f"ENCODING='{enc}'",
                "AUTO_DETECT=FALSE",
                "ALL_VARCHAR=TRUE",
                "SAMPLE_SIZE=100000",
            ]
            opts_str = ",\n                         ".join(opts)
            return f"""
            COPY (
              SELECT *
              FROM read_csv('{input_uri}',
                             {opts_str})
            ) TO '{output_uri}'
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000);
            """

        def make_copy_sql(enc: str | None = None, delim: str | None = None, all_varchar: bool = False) -> str:
            opts = [
                "SAMPLE_SIZE=2000000",
                "HEADER=TRUE",
                "AUTO_DETECT=TRUE",
            ]
            if proceed_on_errors:
                opts.append("IGNORE_ERRORS=TRUE")
            if enc:
                opts.append(f"ENCODING='{enc}'")
            if delim:
                opts.append(f"DELIM='{delim}'")
            if all_varchar:
                opts.append("ALL_VARCHAR=TRUE")
            opts_str = ",\n                         ".join(opts)
            return f"""
            COPY (
              SELECT *
              FROM read_csv_auto('{input_uri}',
                                 {opts_str})
            ) TO '{output_uri}'
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000);
            """

        # Try forced fast path first
        try:
            con.sql(make_copy_sql_forced())
            return
        except Exception:
            logger.info("Forced read_csv path failed; falling back to auto-detect")

        # Try default, then CP1252/ISO8859_1 with pipe then tab, with ALL_VARCHAR fallback
        try:
            con.sql(make_copy_sql())
            return
        except Exception as e1:
            logger.warning(f"Default read_csv_auto failed, trying fallback encodings/delims… ({str(e1)[:220]}…)")

        for delim in ('|', '\t'):
            for enc in ("CP1252", "ISO8859_1"):
                try:
                    con.sql(make_copy_sql(enc=enc, delim=delim))
                    return
                except Exception as e_enc:
                    logger.warning(f"{enc} + '{delim}' failed; trying ALL_VARCHAR… ({str(e_enc)[:160]}…)")
                    try:
                        con.sql(make_copy_sql(enc=enc, delim=delim, all_varchar=True))
                        return
                    except Exception:
                        pass

        # Final fallback
        con.sql(make_copy_sql(enc="CP1252", all_varchar=True))
    finally:
        try:
            con.close()
        except Exception:
            pass
def _load_expected_header(dataset: str) -> List[str]:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if dataset == 'medical':
        path = os.path.join(base_dir, 'apcd', 'medical', 'medical_head.txt')
    else:
        path = os.path.join(base_dir, 'apcd', 'pharmacy', 'pharmacy_head.txt')
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline().rstrip('\n')
    return header.split('|')


def _sanitize_rejects_to_s3(input_uri: str, dataset: str, bronze_root: str, logger: logging.Logger) -> Tuple[str, str, dict]:
    """Stream sanitize a rejects file to S3 fixed; spill unrecoverable lines to rejects2.

    Returns (fixed_uri, rejects2_uri, stats)
    """
    expected_cols = _load_expected_header(dataset)
    n_expected = len(expected_cols)
    tail_numeric = 3 if dataset == 'medical' else 4

    # pick a spill index: nearest 'Desc' or 'Name' column before tail
    head_n = n_expected - tail_numeric
    spill_idx = head_n - 1
    for i in range(head_n - 1, -1, -1):
        name = expected_cols[i].lower()
        if 'desc' in name or 'name' in name:
            spill_idx = i
            break

    in_bucket, in_key = _parse_s3_uri(input_uri)
    dest = bronze_root.replace('s3://', '')
    out_bucket = dest.split('/', 1)[0]
    out_prefix = dest.split('/', 1)[1] if '/' in dest else ''
    base = os.path.basename(in_key)
    # base may be <name>.rejects.txt
    name = base[:-12] if base.lower().endswith('.rejects.txt') else os.path.splitext(base)[0]
    fixed_key = f"{out_prefix}_fixed/{dataset}/{name}.fixed.txt" if out_prefix else f"_fixed/{dataset}/{name}.fixed.txt"
    bad2_key = f"{out_prefix}_rejects2/{dataset}/{name}.rejects2.txt" if out_prefix else f"_rejects2/{dataset}/{name}.rejects2.txt"

    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=in_bucket, Key=in_key)
    stream = obj['Body']

    import io
    fixed_buf = io.BytesIO()
    bad2_buf = io.BytesIO()

    # write header
    fixed_buf.write(('|'.join(expected_cols) + '\n').encode('utf-8'))

    stats = {"total": 0, "fixed": 0, "padded": 0, "spilled": 0, "unfixable": 0}
    for raw in stream.iter_lines(chunk_size=8*1024*1024):
        if not raw:
            continue
        stats["total"] += 1
        # decode permissively
        try:
            line = raw.decode('cp1252', errors='ignore')
        except Exception:
            line = raw.decode('latin1', errors='ignore')
        parts = line.rstrip('\n').split('|')
        m = len(parts)
        if m == n_expected:
            fixed_buf.write((line.rstrip('\n') + '\n').encode('utf-8'))
            stats["fixed"] += 1
            continue
        if m < n_expected:
            parts = parts + [''] * (n_expected - m)
            fixed_buf.write(('|'.join(parts) + '\n').encode('utf-8'))
            stats["padded"] += 1
            continue
        # m > n_expected: right-anchor numeric tail, spill overage into spill_idx
        if m >= tail_numeric:
            tail = parts[-tail_numeric:]
            head = parts[:m - tail_numeric]
            # We need head_n head fields
            if len(head) >= head_n:
                left = head[:spill_idx]
                spill = '|'.join(head[spill_idx:len(head) - (head_n - spill_idx - 1)])
                right = head[len(head) - (head_n - spill_idx - 1):head_n]
                new_head = left + [spill] + right
            else:
                new_head = head + [''] * (head_n - len(head))
            rebuilt = new_head + tail
            if len(rebuilt) == n_expected:
                fixed_buf.write(('|'.join(rebuilt) + '\n').encode('utf-8'))
                stats["spilled"] += 1
                continue
        # if still not correct, send to rejects2
        bad2_buf.write((line.rstrip('\n') + '\n').encode('utf-8'))
        stats["unfixable"] += 1

    # upload
    fixed_buf.seek(0)
    s3.upload_fileobj(fixed_buf, out_bucket, fixed_key)
    bad2_buf.seek(0)
    if bad2_buf.getbuffer().nbytes > 0:
        s3.upload_fileobj(bad2_buf, out_bucket, bad2_key)

    fixed_uri = f"s3://{out_bucket}/{fixed_key}"
    bad2_uri = f"s3://{out_bucket}/{bad2_key}"
    logger.info(f"SANITIZE_DONE {name}: stats={stats} fixed={fixed_uri} rejects2={bad2_uri}")
    return fixed_uri, bad2_uri, stats


def _reprocess_worker(t: Tuple[str, str, str, int, Optional[str], bool, bool, str]) -> Tuple[str, bool, str]:
    """Top-level worker for ProcessPoolExecutor (must be picklable).

    Args tuple: (in_uri, out_uri, dataset, duckdb_threads, tmp_dir, proceed_on_errors, sanitize, bronze_root)
    Returns: (out_uri, success, message)
    """
    in_uri, out_uri, ds, duckdb_threads, tmp_dir, proceed_on_errors, sanitize, bronze_root = t
    logger = setup_logger()
    try:
        out_bucket, out_key = _parse_s3_uri(out_uri)
        if s3_exists(out_bucket, out_key):
            return out_uri, True, "skipped_exists"
        # optional sanitize
        src_uri = in_uri
        if sanitize:
            try:
                fixed_uri, bad2_uri, st = _sanitize_rejects_to_s3(in_uri, ds, bronze_root, logger)
                logger.info(f"Sanitized: {fixed_uri} (unfixable: {st['unfixable']})")
                src_uri = fixed_uri
            except Exception as se:
                return out_uri, False, f"sanitize_error: {se}"
        convert_rejects_to_parquet(src_uri, out_uri, duckdb_threads, tmp_dir, proceed_on_errors, logger)
        if not s3_exists(out_bucket, out_key):
            return out_uri, False, "post_write_missing"
        return out_uri, True, "converted"
    except Exception as e:
        return out_uri, False, f"error: {e}"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Reprocess fixed rejects: convert corrected TXT to Parquet and append as parts")
    ap.add_argument("--dataset", choices=["medical", "pharmacy", "both"], default="both")
    ap.add_argument("--bronze-root", default="s3://pgxdatalake/bronze/")
    ap.add_argument("--fixed-prefix", default=None, help="Optional S3 prefix where corrected rejects live; otherwise use bronze/_rejects/<dataset>/")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--duckdb-threads", type=int, default=1)
    ap.add_argument("--tmp-dir", default=None)
    ap.add_argument("--proceed-on-errors", action="store_true")
    ap.add_argument("--sanitize", action="store_true", help="Auto-fix rejects using known headers; track unfixable lines to _rejects2")
    ap.add_argument("--aggregate-root", default="s3://pgxdatalake/pgx_pipeline/", help="S3 root for aggregated run summaries (for BI)")

    args = ap.parse_args(argv)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    if setup_logging:
        logger, log_buffer = setup_logging("reprocess_txt_to_parquet", args.dataset, run_id)
    else:
        logger = setup_logger()
        log_buffer = []

    try:
        logger.info("🚀 Reprocessing corrected rejects → Parquet parts")
        run_started = time.time()
        agg = {"tx": "repro", "run_id": run_id, "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "appended": 0, "skipped": 0, "errors": 0}
        # Optional pipeline state
        if PipelineState is not None:
            ps = PipelineState('reprocess_txt_to_parquet', f"{args.dataset}_{run_id}", logger)
            ps.mark_step_completed('start', {"workers": args.workers})
        else:
            ps = None

        datasets = [args.dataset] if args.dataset != "both" else ["medical", "pharmacy"]
        total_converted = 0

        tasks: List[Tuple[str, str, str]] = []  # (input_uri, output_uri, dataset)
        s3 = boto3.client("s3")
        for ds in datasets:
            if args.fixed_prefix:
                fixed_bucket, fixed_key = _parse_s3_uri(args.fixed_prefix)
                prefix = f"{fixed_key.rstrip('/')}/{ds}/" if fixed_key else f"{ds}/"
                token = None
                while True:
                    kw = {"Bucket": fixed_bucket, "Prefix": prefix}
                    if token:
                        kw["ContinuationToken"] = token
                    resp = s3.list_objects_v2(**kw)
                    for obj in resp.get("Contents", []):
                        key = obj["Key"]
                        if key.lower().endswith(".txt"):
                            base = os.path.basename(key)
                            name, _ = os.path.splitext(base)
                            # Append part name based on ETag for idempotency
                            et = get_etag(fixed_bucket, key)[:8] or run_id
                            out_bucket, out_root = _parse_s3_uri(args.bronze_root)
                            out_key = f"{out_root.rstrip('/')}/{ds}/{name}.part_{et}.parquet" if out_root else f"{ds}/{name}.part_{et}.parquet"
                            in_uri = f"s3://{fixed_bucket}/{key}"
                            out_uri = f"s3://{out_bucket}/{out_key}"
                            tasks.append((in_uri, out_uri, ds))
                    if resp.get("IsTruncated"):
                        token = resp.get("NextContinuationToken")
                    else:
                        break
            else:
                # Use default rejects location
                for rej in list_reject_files(args.bronze_root, ds):
                    in_bucket, in_key = _parse_s3_uri(rej)
                    base = os.path.basename(in_key)
                    # base looks like <name>.rejects.txt
                    if base.lower().endswith(".rejects.txt"):
                        name = base[:-12]
                    else:
                        name, _ = os.path.splitext(base)
                    et = get_etag(in_bucket, in_key)[:8] or run_id
                    out_bucket, out_root = _parse_s3_uri(args.bronze_root)
                    out_key = f"{out_root.rstrip('/')}/{ds}/{name}.part_{et}.parquet" if out_root else f"{ds}/{name}.part_{et}.parquet"
                    out_uri = f"s3://{out_bucket}/{out_key}"
                    tasks.append((rej, out_uri, ds))

        if not tasks:
            logger.info("Nothing to reprocess (no rejects or fixed files found)")
            return 0

        # Execute (parallel or serial)
        processed = 0
        if args.workers > 1:
            import concurrent.futures as cf
            logger.info(f"Starting parallel reprocess with {args.workers} workers…")
            # Build worker args to avoid capturing non-picklable closures
            worker_args = [
                (in_uri, out_uri, ds, args.duckdb_threads, args.tmp_dir, args.proceed_on_errors, args.sanitize, args.bronze_root)
                for (in_uri, out_uri, ds) in tasks
            ]
            with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
                for out_uri, ok, msg in ex.map(_reprocess_worker, worker_args):
                    if ok:
                        if msg == "converted":
                            logger.info(f"✓ {out_uri}")
                            processed += 1
                            agg["appended"] += 1

                        else:
                            logger.info(f"↷ {msg}: {out_uri}")
                            if msg == "skipped_exists":
                                agg["skipped"] += 1
                    else:
                        logger.error(f"✗ {out_uri} -> {msg}")
                        agg["errors"] += 1
        else:
            for in_uri, out_uri, ds in tasks:
                try:
                    out_bucket, out_key = _parse_s3_uri(out_uri)
                    if s3_exists(out_bucket, out_key):
                        logger.info(f"↷ skipped_exists: {out_uri}")
                        agg["skipped"] += 1
                        continue
                    src_uri = in_uri
                    if args.sanitize:
                        try:
                            fixed_uri, bad2_uri, st = _sanitize_rejects_to_s3(in_uri, ds, args.bronze_root, logger)
                            logger.info(f"Sanitized: {fixed_uri} (unfixable: {st['unfixable']})")
                            src_uri = fixed_uri
                        except Exception as se:
                            agg["errors"] += 1
                            logger.error(f"sanitize_error: {se}")
                            continue
                    convert_rejects_to_parquet(src_uri, out_uri, args.duckdb_threads, args.tmp_dir, args.proceed_on_errors, logger)
                    if not s3_exists(out_bucket, out_key):
                        logger.error(f"post_write_missing: {out_uri}")
                        agg["errors"] += 1
                        continue
                    logger.info(f"✓ {out_uri}")
                    processed += 1
                    agg["appended"] += 1

                except Exception as e:
                    logger.error(f"✗ {out_uri} -> error: {e}")
                    agg["errors"] += 1

        total_converted += processed
        # Close out status
        run_finished = time.time()
        agg["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(run_finished))
        agg["duration_sec"] = round(run_finished - run_started, 3)
        if agg["errors"] > 0:
            agg["status"] = "error"
            agg["status_code"] = 1
        else:
            agg["status"] = "success"
            agg["status_code"] = 0

        logger.info(f"✅ Reprocess complete. Files appended: {total_converted}")
        if save_logs_to_s3 and setup_logging:
            try:
                save_logs_to_s3(log_buffer, "reprocess_txt_to_parquet", args.dataset, run_id, "apcd_input_data", logger=logger)
            except Exception:
                pass
        # Aggregate summary to pgxdatalake
        try:
            agg_bucket, agg_key_root = args.aggregate_root.replace("s3://", "").split("/", 1)
        except ValueError:
            agg_bucket, agg_key_root = args.aggregate_root.replace("s3://", ""), ""
        agg_key = f"{agg_key_root.rstrip('/')}/reprocess_txt_to_parquet/run_id={run_id}/summary.json" if agg_key_root else f"reprocess_txt_to_parquet/run_id={run_id}/summary.json"
        try:
            boto3.client('s3').put_object(Bucket=agg_bucket, Key=agg_key, Body=json.dumps(agg, indent=2).encode('utf-8'), ContentType='application/json')
            logger.info(f"📊 Aggregated summary saved: s3://{agg_bucket}/{agg_key}")
        except Exception as e:
            logger.warning(f"⚠️ Could not save aggregated summary: {e}")
        if 'ps' in locals() and ps:
            ps.mark_pipeline_completed({"appended": total_converted, "errors": agg["errors"]})
        return 0
    except Exception as e:
        logger.error(f"❌ Reprocess failed: {e}")
        if save_logs_to_s3 and setup_logging:
            try:
                save_logs_to_s3(log_buffer, "reprocess_txt_to_parquet", args.dataset, run_id, "apcd_input_data", logger=logger)
            except Exception:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())


