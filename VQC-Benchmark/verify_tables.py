"""
Comprehensive verification script: Cross-check every numerical value in paper.tex tables
against the CSV source data files.
Updated for dual-dataset tables with multirow format.
"""
import csv
import re
import os

RESULTS_DIR = r"C:\Users\user\vqc-benchmark\results"
PAPER_PATH = r"C:\Users\user\vqc-benchmark\paper.tex"

def read_csv(filename):
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def fmt3(val):
    return f"{float(val):.3f}"

def compare_values(expected, found, label, tolerance=0.0015):
    try:
        e = float(expected)
        f = float(found)
    except (ValueError, TypeError):
        return False, f"  MISMATCH [{label}]: expected={expected}, found={found} (non-numeric)"
    if abs(e - f) <= tolerance:
        return True, None
    else:
        return False, f"  MISMATCH [{label}]: CSV={e:.4f}, paper={found}"

def read_paper():
    with open(PAPER_PATH, 'r', encoding='utf-8') as f:
        return f.read()

def extract_table(paper, label):
    m = re.search(rf'\\label\{{{label}\}}(.*?)\\end\{{tabular\}}', paper, re.DOTALL)
    return m.group(1) if m else ""


# ============================================================
# TABLE II: test accuracy (8 models x 3 datasets)
# ============================================================
def verify_table2():
    print("=" * 70)
    print("TABLE II (tab:accuracy): Test accuracy comparison (10 seeds)")
    print("=" * 70)

    data = read_csv("experiment1_model_comparison.csv")
    paper = read_paper()
    table_text = extract_table(paper, "tab:accuracy")

    expected = {}
    for row in data:
        expected[(row['model'], row['dataset'])] = (float(row['test_accuracy']), float(row['test_accuracy_std']))

    paper_models = ['SVM', 'MLP', 'RF', 'VQC-Angle-RA', 'VQC-Angle-ESU2',
                    'VQC-Angle-RA-reps3', 'VQC-ZZ-RA', 'VQC-ZZ-ESU2']
    datasets_order = ['Iris', 'Wine', 'Breast Cancer']

    total_checks = 0
    mismatches = 0
    mismatch_details = []

    for line in table_text.split('\n'):
        line = line.strip()
        clean = re.sub(r'\\textbf\{([^}]*)\}', r'\1', line)

        found_model = None
        for m in paper_models:
            if clean.startswith(m + ' ') or clean.startswith(m + '&') or clean == m:
                found_model = m
                break

        if not found_model:
            continue

        values = re.findall(r'\.(\d{3})\$\\pm\$\.(\d{3})', line)
        if len(values) != 3:
            continue

        for i, ds in enumerate(datasets_order):
            acc_paper = f"0.{values[i][0]}"
            std_paper = f"0.{values[i][1]}"
            csv_acc, csv_std = expected[(found_model, ds)]

            total_checks += 1
            ok, d = compare_values(csv_acc, acc_paper, f"{found_model}/{ds}/acc")
            if not ok: mismatches += 1; mismatch_details.append(d)

            total_checks += 1
            ok, d = compare_values(csv_std, std_paper, f"{found_model}/{ds}/std")
            if not ok: mismatches += 1; mismatch_details.append(d)

    print(f"  Values checked: {total_checks}, Mismatches: {mismatches}")
    for d in mismatch_details: print(d)
    print()
    return total_checks, mismatches, mismatch_details


# ============================================================
# TABLE: QSVC comparison
# ============================================================
def verify_qsvc():
    print("=" * 70)
    print("TABLE (tab:qsvc): QSVC vs VQC vs Classical SVM")
    print("=" * 70)

    data_qsvc = read_csv("experiment7_qsvc.csv")
    data_exp1 = read_csv("experiment1_model_comparison.csv")
    paper = read_paper()
    table_text = extract_table(paper, "tab:qsvc")

    expected = {}
    for row in data_qsvc:
        expected[(row['model'], row['dataset'])] = (float(row['test_accuracy']), float(row['test_accuracy_std']))
    for row in data_exp1:
        if row['model'] == 'VQC-Angle-RA':
            expected[('VQC-Angle-RA', row['dataset'])] = (float(row['test_accuracy']), float(row['test_accuracy_std']))
        if row['model'] == 'SVM':
            expected[('SVM (Classical)', row['dataset'])] = (float(row['test_accuracy']), float(row['test_accuracy_std']))

    paper_models = ['QSVC-Angle', 'QSVC-ZZ', 'VQC-Angle-RA', 'SVM (Classical)']
    datasets_order = ['Iris', 'Wine', 'Breast Cancer']

    total_checks = 0
    mismatches = 0
    mismatch_details = []

    for line in table_text.split('\n'):
        line = line.strip()
        clean = re.sub(r'\\textbf\{([^}]*)\}', r'\1', line)

        found_model = None
        for m in paper_models:
            if clean.startswith(m):
                found_model = m
                break
        if not found_model:
            continue

        values = re.findall(r'\.(\d{3})\$\\pm\$\.(\d{3})', line)
        if len(values) != 3:
            continue

        for i, ds in enumerate(datasets_order):
            acc_paper = f"0.{values[i][0]}"
            std_paper = f"0.{values[i][1]}"
            csv_acc, csv_std = expected[(found_model, ds)]

            total_checks += 1
            ok, d = compare_values(csv_acc, acc_paper, f"{found_model}/{ds}/acc")
            if not ok: mismatches += 1; mismatch_details.append(d)

            total_checks += 1
            ok, d = compare_values(csv_std, std_paper, f"{found_model}/{ds}/std")
            if not ok: mismatches += 1; mismatch_details.append(d)

    print(f"  Values checked: {total_checks}, Mismatches: {mismatches}")
    for d in mismatch_details: print(d)
    print()
    return total_checks, mismatches, mismatch_details


# ============================================================
# Multirow table helpers
# ============================================================
DS_MAP = {'Iris': 'Iris', 'BC': 'Breast Cancer'}

def _check(expected, found, label, results):
    """Append check result to results list: (total_inc, mismatch_inc, detail_or_None)."""
    ok, d = compare_values(expected, found, label)
    results.append((1, 0 if ok else 1, d))

def _run_checks(results, table_name):
    total = sum(r[0] for r in results)
    mis = sum(r[1] for r in results)
    details = [r[2] for r in results if r[2]]
    print("=" * 70)
    print(table_name)
    print("=" * 70)
    print(f"  Values checked: {total}, Mismatches: {mis}")
    for d in details: print(d)
    print()
    return total, mis, details


# ============================================================
# TABLE: Data size (4 models per row, .NNN format)
# ============================================================
def verify_datasize():
    data = read_csv("experiment2_data_size.csv")
    paper = read_paper()
    table_text = extract_table(paper, "tab:datasize")

    expected = {}
    for row in data:
        expected[(row['dataset'], float(row['fraction']), row['model'])] = (
            float(row['test_accuracy']), float(row['test_accuracy_std']))

    models_order = ['VQC', 'SVM', 'MLP', 'RF']
    results = []
    current_ds = None

    for line in table_text.split('\n'):
        line = line.strip()
        ds_match = re.search(r'\\multirow\{\d+\}\{\*\}\{([^}]+)\}', line)
        if ds_match:
            current_ds = ds_match.group(1)
        if not current_ds:
            continue
        frac_match = re.search(r'(\d+)\\%', line)
        if not frac_match:
            continue
        frac = int(frac_match.group(1)) / 100.0
        pairs = re.findall(r'\.(\d{3})\$\\pm\$\.(\d{3})', line)
        if len(pairs) != 4:
            continue
        csv_ds = DS_MAP.get(current_ds, current_ds)
        for i, model in enumerate(models_order):
            acc_p, std_p = f"0.{pairs[i][0]}", f"0.{pairs[i][1]}"
            csv_acc, csv_std = expected[(csv_ds, frac, model)]
            _check(csv_acc, acc_p, f"{csv_ds}/{frac}/{model}/acc", results)
            _check(csv_std, std_p, f"{csv_ds}/{frac}/{model}/std", results)

    return _run_checks(results, "TABLE (tab:datasize): Data size impact")


# ============================================================
# TABLE: Optimizer comparison (0.NNN format, 1 acc pair per row)
# ============================================================
def verify_optimizer():
    data = read_csv("experiment3_optimizer.csv")
    paper = read_paper()
    table_text = extract_table(paper, "tab:optimizer")

    expected = {}
    for row in data:
        expected[(row['dataset'], row['optimizer'])] = (
            float(row['test_accuracy']), float(row['test_accuracy_std']))

    results = []
    current_ds = None

    for line in table_text.split('\n'):
        line = line.strip()
        ds_match = re.search(r'\\multirow\{\d+\}\{\*\}\{([^}]+)\}', line)
        if ds_match:
            current_ds = ds_match.group(1)
        if not current_ds:
            continue
        opt_match = re.search(r'(COBYLA|SPSA)', line)
        if not opt_match:
            continue
        opt = opt_match.group(1).lower()
        pairs = re.findall(r'(\d+\.\d+)\$\\pm\$(\d+\.\d+)', line)
        if not pairs:
            continue
        csv_ds = DS_MAP.get(current_ds, current_ds)
        csv_acc, csv_std = expected[(csv_ds, opt)]
        _check(csv_acc, pairs[0][0], f"{csv_ds}/{opt}/acc", results)
        _check(csv_std, pairs[0][1], f"{csv_ds}/{opt}/std", results)

    return _run_checks(results, "TABLE (tab:optimizer): Optimizer comparison")


# ============================================================
# TABLE: Circuit depth (0.NNN format, 1 acc pair per row)
# ============================================================
def verify_depth():
    data = read_csv("experiment4_circuit_depth.csv")
    paper = read_paper()
    table_text = extract_table(paper, "tab:depth")

    expected = {}
    for row in data:
        expected[(row['dataset'], int(row['reps']))] = (
            float(row['test_accuracy']), float(row['test_accuracy_std']))

    results = []
    current_ds = None

    for line in table_text.split('\n'):
        line = line.strip()
        clean = re.sub(r'\\textbf\{([^}]*)\}', r'\1', line)
        ds_match = re.search(r'\\multirow\{\d+\}\{\*\}\{([^}]+)\}', clean)
        if ds_match:
            current_ds = ds_match.group(1)
        if not current_ds:
            continue
        reps_match = re.search(r'&\s*(\d+)\s*&', clean)
        if not reps_match:
            continue
        reps = int(reps_match.group(1))
        pairs = re.findall(r'(\d+\.\d+)\$\\pm\$(\d+\.\d+)', clean)
        if not pairs:
            continue
        csv_ds = DS_MAP.get(current_ds, current_ds)
        csv_acc, csv_std = expected[(csv_ds, reps)]
        _check(csv_acc, pairs[0][0], f"{csv_ds}/reps{reps}/acc", results)
        _check(csv_std, pairs[0][1], f"{csv_ds}/reps{reps}/std", results)

    return _run_checks(results, "TABLE (tab:depth): Circuit depth")


# ============================================================
# TABLE: Noise impact (.NNN format, 2 value pairs per row)
# ============================================================
def verify_noise():
    data = read_csv("experiment5_noise.csv")
    paper = read_paper()
    table_text = extract_table(paper, "tab:noise")

    expected = {}
    for row in data:
        expected[(row['dataset'], float(row['error_rate']))] = {
            'acc': float(row['test_accuracy']), 'acc_std': float(row['test_accuracy_std']),
            'f1': float(row['test_f1']), 'f1_std': float(row['test_f1_std'])}

    results = []
    current_ds = None

    for line in table_text.split('\n'):
        line = line.strip()
        clean = re.sub(r'\\textbf\{([^}]*)\}', r'\1', line)
        ds_match = re.search(r'\\multirow\{\d+\}\{\*\}\{([^}]+)\}', clean)
        if ds_match:
            current_ds = ds_match.group(1)
        if not current_ds:
            continue
        if 'Ideal' in clean:
            err = 0.0
        else:
            err_match = re.search(r'p=(\d+\.?\d*)', clean)
            if not err_match:
                continue
            err = float(err_match.group(1))
        pairs = re.findall(r'\.(\d{3})\$\\pm\$\.(\d{3})', clean)
        if len(pairs) < 2:
            continue
        csv_ds = DS_MAP.get(current_ds, current_ds)
        vals = expected[(csv_ds, err)]
        _check(vals['acc'], f"0.{pairs[0][0]}", f"{csv_ds}/p={err}/acc", results)
        _check(vals['acc_std'], f"0.{pairs[0][1]}", f"{csv_ds}/p={err}/acc_std", results)
        _check(vals['f1'], f"0.{pairs[1][0]}", f"{csv_ds}/p={err}/f1", results)
        _check(vals['f1_std'], f"0.{pairs[1][1]}", f"{csv_ds}/p={err}/f1_std", results)

    return _run_checks(results, "TABLE (tab:noise): Noise impact")


# ============================================================
# TABLE: Convergence (0.NNN format, 1 acc pair per row)
# ============================================================
def verify_convergence():
    data = read_csv("experiment6_convergence.csv")
    paper = read_paper()
    table_text = extract_table(paper, "tab:convergence")

    expected = {}
    for row in data:
        expected[(row['dataset'], int(row['maxiter']))] = (
            float(row['test_accuracy']), float(row['test_accuracy_std']))

    results = []
    current_ds = None

    for line in table_text.split('\n'):
        line = line.strip()
        ds_match = re.search(r'\\multirow\{\d+\}\{\*\}\{([^}]+)\}', line)
        if ds_match:
            current_ds = ds_match.group(1)
        if not current_ds:
            continue
        iter_match = re.search(r'&\s*(\d+)\s*&', line)
        if not iter_match:
            continue
        maxiter = int(iter_match.group(1))
        pairs = re.findall(r'(\d+\.\d+)\$\\pm\$(\d+\.\d+)', line)
        if not pairs:
            continue
        csv_ds = DS_MAP.get(current_ds, current_ds)
        csv_acc, csv_std = expected[(csv_ds, maxiter)]
        _check(csv_acc, pairs[0][0], f"{csv_ds}/maxiter{maxiter}/acc", results)
        _check(csv_std, pairs[0][1], f"{csv_ds}/maxiter{maxiter}/std", results)

    return _run_checks(results, "TABLE (tab:convergence): Convergence")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("COMPREHENSIVE VERIFICATION: paper.tex vs CSV source data")
    print("=" * 70 + "\n")

    results = []

    # Table II: Model comparison (standard 3-column format)
    results.append(("Table II (accuracy)", *verify_table2()))

    # Table QSVC
    results.append(("Table QSVC", *verify_qsvc()))

    # Multirow tables (dual-dataset)
    results.append(("Table Datasize", *verify_datasize()))
    results.append(("Table Optimizer", *verify_optimizer()))
    results.append(("Table Depth", *verify_depth()))
    results.append(("Table Noise", *verify_noise()))
    results.append(("Table Convergence", *verify_convergence()))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    grand_total = 0
    grand_mismatches = 0
    all_details = []

    for name, total, mis, details in results:
        print(f"  {name}: {total} values checked, {mis} mismatches")
        grand_total += total
        grand_mismatches += mis
        all_details.extend(details)

    print(f"\n  GRAND TOTAL: {grand_total} values checked, {grand_mismatches} mismatches")

    if grand_mismatches > 0:
        print("\n  ALL MISMATCHES:")
        for d in all_details:
            print(d)
    else:
        print("\n  ALL VALUES VERIFIED CORRECTLY - NO MISMATCHES FOUND")

    print()
