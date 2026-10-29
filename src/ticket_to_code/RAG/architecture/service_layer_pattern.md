# Service Layer Implementation Pattern

## Standard Service Structure

All services in ContentBridge follow this pattern:

```csharp
using Microsoft.Extensions.Logging;
using ContentBridge.Interfaces;
using ContentBridge.Models;

namespace ContentBridge.Services
{
    public class ExampleService : IExampleService
    {
        private readonly ILogger<ExampleService> _logger;
        private readonly IRepository<Entity> _repository;
        
        // Constructor: Use dependency injection
        public ExampleService(
            ILogger<ExampleService> logger,
            IRepository<Entity> repository)
        {
            _logger = logger;
            _repository = repository;
        }
        
        // Public methods: Implement interface
        public async Task<Result> DoSomethingAsync(Request request)
        {
            // 1. Validate input
            ValidateInput(request);
            
            // 2. Log operation
            _logger.LogInformation($"Processing: {request.Id}");
            
            try
            {
                // 3. Business logic
                var result = await ProcessAsync(request);
                
                // 4. Log success
                _logger.LogInformation($"Success: {result.Id}");
                
                return result;
            }
            catch (Exception ex)
            {
                // 5. Log error
                _logger.LogError(ex, $"Failed: {request.Id}");
                throw;
            }
        }
        
        // Private methods: Helper methods
        private void ValidateInput(Request request)
        {
            if (request == null)
                throw new ArgumentNullException(nameof(request));
        }
    }
}
```

## Dependency Injection Registration

```csharp
// In Startup.cs or Program.cs
services.AddScoped<IExampleService, ExampleService>();
```

## Repository Pattern

All data access uses IRepository<T>:

```csharp
// Read
var entity = await _repository.GetByIdAsync(id);
var entities = await _repository.GetAllAsync();

// Write
await _repository.AddAsync(entity);
await _repository.UpdateAsync(entity);
await _repository.DeleteAsync(id);
```

## Controller Integration

```csharp
[ApiController]
[Route("api/[controller]")]
public class ExampleController : ControllerBase
{
    private readonly IExampleService _service;
    
    public ExampleController(IExampleService service)
    {
        _service = service;
    }
    
    [HttpPost]
    public async Task<IActionResult> Create([FromBody] Request request)
    {
        var result = await _service.DoSomethingAsync(request);
        return Ok(result);
    }
}
```
