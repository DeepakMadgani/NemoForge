import pytest
from ticket_to_code.runtime.diagnostic_error_slicer import DiagnosticErrorSlicer


def test_slice_java_maven_compiler_error():
    """Verify Maven [ERROR] javac blocks preserve symbol and location context lines."""
    raw_maven_output = """
[INFO] Scanning for projects...
[INFO] -------------------< com.example:user-service >-------------------
[INFO] Building user-service 1.0.0
[INFO] --- maven-compiler-plugin:3.8.1:compile (default-compile) @ user-service ---
[INFO] Changes detected - recompiling the module!
[INFO] Compiling 42 source files to C:\\project\\target\\classes
[ERROR] /src/main/java/com/example/UserService.java:[34,18] cannot find symbol
[ERROR]   symbol:   variable userToken
[ERROR]   location: class com.example.UserService
[ERROR] /src/main/java/com/example/UserService.java:[50,22] incompatible types: java.lang.String cannot be converted to int
[INFO] 2 errors 
[INFO] ------------------------------------------------------------------------
[INFO] BUILD FAILURE
[INFO] ------------------------------------------------------------------------
[INFO] Total time:  2.345 s
[INFO] Finished at: 2026-10-14T12:00:00Z
"""
    sliced = DiagnosticErrorSlicer.slice_errors(raw_maven_output, max_errors=2)
    assert "/src/main/java/com/example/UserService.java:34" in sliced
    assert "cannot find symbol" in sliced
    assert "symbol:   variable userToken" in sliced
    assert "location: class com.example.UserService" in sliced
    assert "incompatible types" in sliced
    # Ensure Maven reactor fluff is stripped
    assert "[INFO]" not in sliced
    assert "BUILD FAILURE" not in sliced
    assert "Total time" not in sliced


def test_slice_typescript_compiler_error():
    """Verify TypeScript tsc output is parsed cleanly."""
    raw_tsc_output = """
> frontend@1.0.0 build
> tsc

src/components/UserGrid.tsx(24,15): error TS2339: Property 'isAdmin' does not exist on type 'User'.
src/services/api.ts(12,9): error TS2322: Type 'string' is not assignable to type 'number'.
npm error code 1
npm error A complete log of this run can be found in:
"""
    sliced = DiagnosticErrorSlicer.slice_errors(raw_tsc_output)
    assert "src/components/UserGrid.tsx:24" in sliced
    assert "Property 'isAdmin' does not exist on type 'User'" in sliced
    assert "src/services/api.ts:12" in sliced
    assert "npm error" not in sliced


def test_slice_python_traceback():
    """Verify Python traceback extraction."""
    raw_py_output = """
Traceback (most recent call last):
  File "src/auth/service.py", line 42, in login
    return self.hasher.verify(password, hash)
TypeError: verify() missing 1 required positional argument: 'salt'
"""
    sliced = DiagnosticErrorSlicer.slice_errors(raw_py_output)
    assert "src/auth/service.py:42" in sliced
    assert "TypeError: verify() missing 1 required positional argument: 'salt'" in sliced


def test_slice_csharp_dotnet_error():
    """Verify C# MSBuild / dotnet build compiler error parsing."""
    raw_dotnet_output = """
MSBuild version 17.8.3+195e4f5a3 for .NET
  Determining projects to restore...
  Restored C:\\Project\\App.csproj (in 120 ms).
C:\\Project\\Controllers\\UserController.cs(18,25): error CS0103: The name 'userId' does not exist in the current context [C:\\Project\\App.csproj]

Build FAILED.
    0 Warning(s)
    1 Error(s)
"""
    sliced = DiagnosticErrorSlicer.slice_errors(raw_dotnet_output)
    assert "UserController.cs:18" in sliced
    assert "The name 'userId' does not exist in the current context" in sliced
    assert "MSBuild version" not in sliced
    assert "Determining projects" not in sliced


def test_strict_token_budget_cap():
    """Verify large 10,000-line spam output is strictly capped under budget."""
    spam_error = "\n".join(
        f"[ERROR] /src/File{i}.java:[{i},1] error: syntax error in class {i}" for i in range(100)
    )
    sliced = DiagnosticErrorSlicer.slice_errors(spam_error, max_errors=5, max_tokens_budget=200)
    # Output must be strictly bounded under token budget (~4 chars per token)
    assert len(sliced) <= 200 * 5
    assert "[... additional compiler warnings truncated ...]" in sliced or len(sliced.split("\n\n")) <= 5
