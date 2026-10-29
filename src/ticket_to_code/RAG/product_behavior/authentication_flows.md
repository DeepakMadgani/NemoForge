# Authentication Flow Behavior

## Expected Behavior

### Successful Login
- User enters valid credentials
- System validates against OTDS
- Returns authentication token
- Redirects to dashboard

### Failed Login
- User enters invalid credentials
- System shows error: "Invalid username or password"
- NO token returned
- User remains on login page

### Expired Session
- User token expires after 30 minutes
- System redirects to login
- Shows message: "Session expired. Please login again."

## Business Rules

1. Passwords are case-sensitive
2. Usernames are case-INSENSITIVE
3. After 3 failed attempts: account locked for 15 minutes
4. Password must be at least 8 characters

## Validation Rules

| Field | Rule | Error Message |
|-------|------|---------------|
| Username | Not empty | "Username is required" |
| Username | Min 3 chars | "Username must be at least 3 characters" |
| Password | Not empty | "Password is required" |
| Password | Min 8 chars | "Password must be at least 8 characters" |

## Error Handling Patterns

```csharp
// Example: How authentication errors are handled
public AuthResult Authenticate(string username, string password)
{
    if (string.IsNullOrEmpty(username))
        throw new ArgumentException("Username is required");
    
    if (string.IsNullOrEmpty(password))
        throw new ArgumentException("Password is required");
    
    // OTDS call...
    
    if (!otdsResponse.Success)
        throw new AuthenticationException("Invalid username or password");
}
```

## Test Scenarios That Should Exist

```csharp
[Fact] void ValidCredentials_ReturnsToken()
[Fact] void InvalidCredentials_ThrowsAuthenticationException()
[Fact] void EmptyUsername_ThrowsArgumentException()
[Fact] void EmptyPassword_ThrowsArgumentException()
[Fact] void ExpiredToken_RedirectsToLogin()
```
