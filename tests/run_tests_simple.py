import importlib
import inspect
import sys

from pathlib import Path
import importlib.util

root = Path(__file__).resolve().parent
modules = [
    root / 'test_data_preprocessor.py',
    root / 'test_model_trainer.py',
]
import sys
# Ensure project root is on sys.path so 'src' package can be imported
sys.path.insert(0, str(root.parent))

failed = 0
for path in modules:
    print('\nRunning', path.name)
    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        print('ERROR importing', path.name, type(e).__name__, e)
        failed += 1
        continue
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith('test_'):
            print(' -', name)
            try:
                fn()
                print('   OK')
            except Exception as e:
                print('   FAIL:', type(e).__name__, e)
                failed += 1

if failed:
    print('\n%d test(s) failed' % failed)
    sys.exit(1)
else:
    print('\nAll tests passed')
    sys.exit(0)
