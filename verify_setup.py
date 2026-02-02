"""
Vehicle Classifier - Setup Verification Script

This script verifies that all necessary modules and dependencies are installed
and that the application is ready to run.
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print("✓ Python version OK:", f"{version.major}.{version.minor}")
        return True
    else:
        print("✗ Python 3.10+ required, found:", f"{version.major}.{version.minor}")
        return False


def check_imports():
    """Check all required packages are installed."""
    packages = {
        'fastapi': 'FastAPI',
        'tensorflow': 'TensorFlow',
        'redis': 'Redis',
        'pytest': 'pytest',
        'prometheus_client': 'Prometheus',
        'jose': 'python-jose',
        'passlib': 'passlib',
        'bcrypt': 'bcrypt',
    }
    
    results = []
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"✓ {name} installed")
            results.append(True)
        except ImportError:
            print(f"✗ {name} NOT installed")
            results.append(False)
    
    return all(results)


def check_project_structure():
    """Check project structure."""
    required_dirs = [
        'src',
        'src/api',
        'src/core',
        'tests',
        'logs',
        'uploads',
        'db',
    ]
    
    required_files = [
        'requirements.txt',
        'Dockerfile',
        'docker-compose.yml',
        'README.md',
        'SECURITY.md',
        'DEPLOYMENT.md',
    ]
    
    results = []
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ Directory exists: {dir_path}")
            results.append(True)
        else:
            print(f"✗ Missing directory: {dir_path}")
            results.append(False)
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ File exists: {file_path}")
            results.append(True)
        else:
            print(f"✗ Missing file: {file_path}")
            results.append(False)
    
    return all(results)


def check_core_modules():
    """Check core modules exist."""
    core_modules = [
        'src/core/__init__.py',
        'src/core/security.py',
        'src/core/errors.py',
        'src/core/monitoring.py',
        'src/core/database.py',
        'src/core/redis_client.py',
    ]
    
    results = []
    for module_path in core_modules:
        if Path(module_path).exists():
            print(f"✓ Core module exists: {module_path}")
            results.append(True)
        else:
            print(f"✗ Missing core module: {module_path}")
            results.append(False)
    
    return all(results)


def check_tests():
    """Check test files exist."""
    test_files = [
        'tests/__init__.py',
        'tests/conftest.py',
        'tests/test_api.py',
        'tests/test_auth.py',
        'tests/test_security.py',
        'tests/test_monitoring.py',
    ]
    
    results = []
    for test_file in test_files:
        if Path(test_file).exists():
            print(f"✓ Test file exists: {test_file}")
            results.append(True)
        else:
            print(f"✗ Missing test file: {test_file}")
            results.append(False)
    
    return all(results)


def check_api_module():
    """Check API module structure."""
    api_files = [
        'src/api/app.py',
        'src/api/auth.py',
        'src/api/service.py',
        'src/api/logging_config.py',
    ]
    
    results = []
    for api_file in api_files:
        if Path(api_file).exists():
            print(f"✓ API file exists: {api_file}")
            results.append(True)
        else:
            print(f"✗ Missing API file: {api_file}")
            results.append(False)
    
    return all(results)


def run_health_check():
    """Run health check on FastAPI app."""
    try:
        from src.api.app import app
        print("✓ FastAPI app imports successfully")
        return True
    except Exception as e:
        print(f"✗ FastAPI app import failed: {e}")
        return False


def main():
    """Run all checks."""
    print("=" * 60)
    print("Vehicle Classifier - Setup Verification")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Project Structure", check_project_structure),
        ("Core Modules", check_core_modules),
        ("Test Files", check_tests),
        ("API Module", check_api_module),
        ("Dependencies", check_imports),
        ("FastAPI App", run_health_check),
    ]
    
    results = {}
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 60)
        results[name] = check_func()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} checks passed")
    print("=" * 60)
    
    if passed == total:
        print("\n✓ All checks passed! Ready to deploy.")
        print("\nNext steps:")
        print("1. Set environment variables (see DEPLOYMENT.md)")
        print("2. Run tests: pytest tests/ -v")
        print("3. Start API: uvicorn src.api.app:app --reload")
        print("4. Or use Docker: docker-compose up -d")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        print("\nFor help, see:")
        print("- README.md for overview")
        print("- SECURITY.md for authentication setup")
        print("- DEPLOYMENT.md for deployment instructions")
        return 1


if __name__ == "__main__":
    sys.exit(main())
