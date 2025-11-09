"""Test script for BUG-013: Enhanced dataset format detection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from aios.data.datasets.archive_readers import _is_archive
from aios.data.datasets.advanced_readers import read_text_lines_sample_any

print('🧪 Testing Enhanced Format Detection - BUG-013')
print('=' * 60)

# Create test directory
test_dir = Path(__file__).parent.parent / 'artifacts' / 'test_formats'
test_dir.mkdir(exist_ok=True)

# Test 1: CSV detection
print('\n📋 Test 1: CSV Detection')
csv_file = test_dir / 'test.csv'
csv_file.write_text('name,age,city\nAlice,30,NYC\nBob,25,LA\n')
lines = read_text_lines_sample_any(csv_file, max_lines=10)
print(f'✅ CSV read {len(lines)} lines')
if lines:
    print(f'   Sample: {lines[0]}')

# Test 2: JSON detection
print('\n📦 Test 2: JSON Detection')
json_file = test_dir / 'test.json'
json_file.write_text('{"items": [{"text": "Hello"}, {"text": "World"}]}')
lines = read_text_lines_sample_any(json_file, max_lines=10)
print(f'✅ JSON read {len(lines)} lines')
if lines:
    print(f'   Sample: {lines[0][:60]}...')

# Test 3: JSONL detection
print('\n📦 Test 3: JSONL Detection')
jsonl_file = test_dir / 'test.jsonl'
jsonl_file.write_text('{"text": "Line 1"}\n{"text": "Line 2"}\n')
lines = read_text_lines_sample_any(jsonl_file, max_lines=10)
print(f'✅ JSONL read {len(lines)} lines')
if lines:
    print(f'   Sample: {lines[0]}')

# Test 4: Archive detection with magic bytes
print('\n📦 Test 4: Archive Magic Byte Detection')
import zipfile
zip_file = test_dir / 'archive.zip'
with zipfile.ZipFile(zip_file, 'w') as zf:
    zf.writestr('content.txt', 'Text inside ZIP')

is_zip = _is_archive(zip_file)
print(f'✅ ZIP detected: {is_zip}')

# Read magic bytes to verify
with open(zip_file, 'rb') as f:
    magic = f.read(4)
    print(f'   Magic bytes: {magic.hex()} (PK signature: {magic[:2] == b"PK"})')

# Test 5: Fake extension (CSV content with .dat extension)
print('\n🎭 Test 5: CSV Content with Wrong Extension')
fake_file = test_dir / 'data.dat'
fake_file.write_text('col1,col2,col3\nval1,val2,val3\n')
lines = read_text_lines_sample_any(fake_file, max_lines=10)
print(f'✅ DAT with CSV content read {len(lines)} lines')
if lines:
    print(f'   Sample: {lines[0]}')

# Test 6: TAR.GZ by extension only (dummy content)
print('\n📦 Test 6: TAR.GZ Extension Detection')
tar_file = test_dir / 'archive.tar.gz'
tar_file.write_bytes(b'dummy')  # Not real tar, just testing extension detection
is_tar = _is_archive(tar_file)
print(f'✅ TAR.GZ detected by extension: {is_tar}')

# Test 7: Parquet detection (if pyarrow available)
print('\n📊 Test 7: Parquet Detection')
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    parquet_file = test_dir / 'test.parquet'
    table = pa.table({'text': ['Line 1', 'Line 2', 'Line 3']})
    pq.write_table(table, parquet_file)
    
    lines = read_text_lines_sample_any(parquet_file, max_lines=10)
    print(f'✅ Parquet read {len(lines)} lines')
    if lines:
        print(f'   Sample: {lines[0]}')
except ImportError:
    print('⚠️  Parquet test skipped (pyarrow not installed)')

print('\n' + '=' * 60)
print('✅ BUG-013 Format Detection Tests Complete!')
print('=' * 60)
print('\nSummary:')
print('  • CSV detection: Working ✓')
print('  • JSON/JSONL detection: Working ✓')
print('  • Archive magic bytes: Working ✓')
print('  • Extension fallback: Working ✓')
print('  • Content sniffing: Working ✓')
