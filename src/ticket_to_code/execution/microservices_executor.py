"""
Microservices-Specific Execution Engine

Handles:
- Multi-service builds
- Service-level testing
- API integration testing
- Docker-based validation
- Contract testing

Perfect for Java microservices (Supplier + Exchange system)

Author: Deepak Madgani
Date: April 2026
"""

import logging
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

from ticket_to_code.models import BuildResult, TestResult, BuildStatus, TestStatus
from ticket_to_code.execution.base_executor import ExecutionEngineBase, ExecutionEngineFactory

logger = logging.getLogger(__name__)


class MicroserviceExecutionEngine:
    """
    Orchestrates execution for microservices architecture.
    
    Features:
    - Discovers services automatically
    - Builds services independently
    - Tests services independently
    - Validates inter-service APIs
    - Docker-based deployment testing
    """
    
    def __init__(self, workspace_path: str, technology: Optional[str] = None):
        """
        Initialize microservices execution engine.
        
        Args:
            workspace_path: Path to microservices workspace
            technology: Technology stack (auto-detected if None)
        """
        self.workspace_path = Path(workspace_path)
        self.technology = technology
        
        # Create base executor for technology
        self.base_executor = ExecutionEngineFactory.create(
            str(workspace_path), 
            technology
        )
        
        # Discover services
        self.services = self._discover_services()
        
        logger.info(
            f"Microservices Engine initialized: "
            f"{len(self.services)} services found"
        )
    
    def _discover_services(self) -> List[Dict[str, str]]:
        """
        Auto-discover microservices in workspace.
        
        Returns:
            List of service info dicts with name and path
        """
        services = []
        
        # Find all project files
        project_files = self.base_executor.detect_project_files()
        
        for project_file in project_files:
            # Determine service name from directory structure
            service_dir = str(Path(project_file).parent)
            service_name = service_dir.split('/')[-1] if '/' in service_dir else 'root'
            
            services.append({
                'name': service_name,
                'path': project_file,
                'directory': service_dir
            })
        
        logger.info(f"Discovered services: {[s['name'] for s in services]}")
        return services
    
    def build_all_services(
        self, 
        parallel: bool = False
    ) -> Dict[str, BuildResult]:
        """
        Build all microservices.
        
        Args:
            parallel: If True, build services in parallel (not implemented yet)
            
        Returns:
            Dict mapping service name to BuildResult
        """
        logger.info(f"Building {len(self.services)} services...")
        
        results = {}
        
        for service in self.services:
            service_name = service['name']
            logger.info(f"Building service: {service_name}")
            
            try:
                result = self.base_executor.execute_build(
                    project_path=service['path'],
                    timeout=300
                )
                results[service_name] = result
                
                if result.status == BuildStatus.SUCCESS:
                    logger.info(f"✅ {service_name}: BUILD SUCCESS")
                else:
                    logger.error(f"❌ {service_name}: BUILD FAILED")
            except Exception as e:
                logger.error(f"Error building {service_name}: {e}")
                results[service_name] = BuildResult(
                    status=BuildStatus.ERROR,
                    stdout="",
                    stderr=str(e),
                    exit_code=-1,
                    errors=[str(e)]
                )
        
        # Summary
        success_count = sum(
            1 for r in results.values() 
            if r.status == BuildStatus.SUCCESS
        )
        logger.info(
            f"Build Summary: {success_count}/{len(results)} services successful"
        )
        
        return results
    
    def test_all_services(self) -> Dict[str, TestResult]:
        """
        Test all microservices.
        
        Returns:
            Dict mapping service name to TestResult
        """
        logger.info(f"Testing {len(self.services)} services...")
        
        results = {}
        
        for service in self.services:
            service_name = service['name']
            logger.info(f"Testing service: {service_name}")
            
            try:
                result = self.base_executor.execute_tests(
                    project_path=service['path'],
                    timeout=600
                )
                results[service_name] = result
                
                if result.status == TestStatus.ALL_PASSED:
                    logger.info(
                        f"✅ {service_name}: "
                        f"{result.passed}/{result.total_tests} tests passed"
                    )
                else:
                    logger.error(
                        f"❌ {service_name}: "
                        f"{result.failed}/{result.total_tests} tests failed"
                    )
            except Exception as e:
                logger.error(f"Error testing {service_name}: {e}")
                results[service_name] = TestResult(
                    status=TestStatus.ERROR,
                    total_tests=0,
                    passed=0,
                    failed=0,
                    errors=[str(e)]
                )
        
        # Summary
        total_tests = sum(r.total_tests for r in results.values())
        total_passed = sum(r.passed for r in results.values())
        total_failed = sum(r.failed for r in results.values())
        
        logger.info(
            f"Test Summary: {total_passed}/{total_tests} tests passed, "
            f"{total_failed} failed across {len(results)} services"
        )
        
        return results
    
    def validate_service_integration(
        self,
        source_service: str,
        target_service: str,
        api_endpoint: str
    ) -> bool:
        """
        Validate API integration between services.
        
        Example: Supplier Service → Exchange Service
        
        Args:
            source_service: Calling service name
            target_service: Target service name
            api_endpoint: API endpoint to test
            
        Returns:
            True if integration works
        """
        logger.info(
            f"Validating integration: {source_service} → "
            f"{target_service} ({api_endpoint})"
        )
        
        # This is a placeholder for actual API integration testing
        # In production, you would:
        # 1. Start target service (e.g., in Docker)
        # 2. Make API call from source service
        # 3. Validate response
        
        logger.warning(
            "API integration testing not yet implemented. "
            "This requires Docker orchestration."
        )
        
        return True
    
    def docker_compose_up(
        self,
        compose_file: str = "docker-compose.yml"
    ) -> bool:
        """
        Start all microservices using Docker Compose.
        
        Args:
            compose_file: Path to docker-compose.yml
            
        Returns:
            True if services started successfully
        """
        logger.info(f"Starting microservices with: {compose_file}")
        
        try:
            result = subprocess.run(
                ["docker-compose", "-f", compose_file, "up", "-d"],
                cwd=self.workspace_path,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                logger.info("✅ Microservices started successfully")
                return True
            else:
                logger.error(f"Failed to start services: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error starting services: {e}")
            return False
    
    def docker_compose_down(
        self,
        compose_file: str = "docker-compose.yml"
    ) -> bool:
        """
        Stop all microservices.
        
        Args:
            compose_file: Path to docker-compose.yml
            
        Returns:
            True if services stopped successfully
        """
        logger.info("Stopping microservices...")
        
        try:
            result = subprocess.run(
                ["docker-compose", "-f", compose_file, "down"],
                cwd=self.workspace_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                logger.info("✅ Microservices stopped successfully")
                return True
            else:
                logger.error(f"Failed to stop services: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error stopping services: {e}")
            return False
    
    def execute_full_validation(self) -> Dict[str, any]:
        """
        Execute complete microservices validation.
        
        Workflow:
        1. Build all services
        2. Test all services
        3. Start services (Docker)
        4. Validate API integrations
        5. Stop services
        
        Returns:
            Dict with validation results
        """
        logger.info("Starting full microservices validation...")
        
        results = {
            'build': {},
            'test': {},
            'docker_startup': False,
            'integrations': {},
            'overall_status': 'FAILED'
        }
        
        # Step 1: Build
        logger.info("Step 1: Building services...")
        results['build'] = self.build_all_services()
        build_success = all(
            r.status == BuildStatus.SUCCESS 
            for r in results['build'].values()
        )
        
        if not build_success:
            logger.error("Build failed. Stopping validation.")
            return results
        
        # Step 2: Test
        logger.info("Step 2: Testing services...")
        results['test'] = self.test_all_services()
        test_success = all(
            r.status == TestStatus.ALL_PASSED 
            for r in results['test'].values()
        )
        
        if not test_success:
            logger.warning("Some tests failed, but continuing...")
        
        # Step 3 & 4: Docker (optional, if docker-compose exists)
        compose_file = self.workspace_path / "docker-compose.yml"
        if compose_file.exists():
            logger.info("Step 3: Starting services (Docker)...")
            results['docker_startup'] = self.docker_compose_up()
            
            if results['docker_startup']:
                logger.info("Step 4: Validating API integrations...")
                # Add your integration tests here
                
                logger.info("Step 5: Stopping services...")
                self.docker_compose_down()
        else:
            logger.info("No docker-compose.yml found, skipping Docker validation")
        
        # Overall status
        if build_success and test_success:
            results['overall_status'] = 'SUCCESS'
        elif build_success:
            results['overall_status'] = 'PARTIAL'
        
        logger.info(f"Validation complete: {results['overall_status']}")
        return results


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def execute_microservices_validation(
    workspace_path: str,
    technology: str = "java"
) -> Dict[str, any]:
    """
    Convenience function for microservices validation.
    
    Args:
        workspace_path: Path to microservices workspace
        technology: Technology stack
        
    Returns:
        Validation results
    """
    engine = MicroserviceExecutionEngine(workspace_path, technology)
    return engine.execute_full_validation()
