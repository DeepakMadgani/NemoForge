import pytest
from pathlib import Path
import sys

# Ensure chatbot backend is on path
chatbot_backend_dir = Path(__file__).parent.parent / "chatbot" / "backend"
if str(chatbot_backend_dir) not in sys.path:
    sys.path.insert(0, str(chatbot_backend_dir))

from services.call_chain_tracer import CallChainTracer


def test_call_chain_tracer_traces_coordinator_dialog(tmp_path):
    # Mock workspace structure reflecting true CC4E architecture
    xchange_dir = tmp_path / "xchange-ui" / "src" / "app" / "modules" / "project" / "coordinators" / "delete-coodinator"
    xchange_dir.mkdir(parents=True)
    i18n_dir = tmp_path / "xchange-ui" / "src" / "assets" / "i18n" / "projects"
    i18n_dir.mkdir(parents=True)
    
    # 1. Source i18n file in src/assets/i18n
    i18n_file = i18n_dir / "en.json"
    i18n_file.write_text('{"project.coordinator.delete.half.msg": "will revoke access from all contracts in this project."}', encoding="utf-8")
    
    # 2. Template
    html_file = xchange_dir / "delete-coodinator.component.html"
    html_file.write_text('<p>{{ "project.coordinator.delete.half.msg" | translate }}</p><button (click)="deleteCoordinator()"></button>', encoding="utf-8")
    
    # 3. Component TS with constructor DI and UI plumbing that should be filtered out
    ts_file = xchange_dir / "delete-coodinator.component.ts"
    ts_file.write_text('''
    export class DeleteCoordinatorComponent {
      constructor(
        private memberService: MemberService,
        private modalService: ModalService,
        private notifService: NotificationService
      ) {}

      deleteCoordinator() {
        this.modalService.close();
        this.notifService.notifySuccess("done");
        this.memberService.updateProjectUsers("p-1", {});
      }
    }
    ''', encoding="utf-8")
    
    # 4. Service TS with nested blocks and pipes (testing balanced brace parser)
    service_dir = tmp_path / "xchange-ui" / "src" / "app" / "services"
    service_dir.mkdir(parents=True)
    service_file = service_dir / "member.service.ts"
    service_file.write_text('''
    export class MemberService {
      updateProjectUsers(p: string, m: any) {
        if (p) {
          return this.sagaService.doCommand("updateProjectUser", { p, m }).pipe(
            switchMap((res) => {
              return of(res);
            })
          );
        }
        return null;
      }
    }
    ''', encoding="utf-8")
    
    # 5. Backend Java Saga Configuration
    java_config_dir = tmp_path / "project-service" / "src" / "main" / "java" / "com" / "cloudforge" / "sagas"
    java_config_dir.mkdir(parents=True)
    config_file = java_config_dir / "SagasConfig.java"
    config_file.write_text('''
    public class SagasConfig {
      private final UpdateProjectUserHandler updateProjectUserHandler;
      
      public void configure() {
        step.addStepHandler("updateProjectUser", updateProjectUserHandler);
      }
    }
    ''', encoding="utf-8")

    # 6. Backend Java Saga Handler implementing process()
    handler_file = java_config_dir / "UpdateProjectUserHandler.java"
    handler_file.write_text('''
    public class UpdateProjectUserHandler {
      public void process(SagaState state) {
        memberService.updateMember("p-1", member);
      }
    }
    ''', encoding="utf-8")
    
    tracer = CallChainTracer(str(tmp_path))
    res = tracer.trace("will revoke access from all contracts in this project")
    
    # Assertions on verified trace
    assert len(res["i18n_matches"]) > 0
    assert "src/assets/i18n" in res["i18n_matches"][0]["file"]
    assert res["i18n_matches"][0]["key"] == "project.coordinator.delete.half.msg"
    
    # Assert UI plumbing (modalService, notifService) is excluded
    svc_fields = [sc["service_field"] for sc in res["service_calls"]]
    assert "memberService" in svc_fields
    assert "modalService" not in svc_fields
    assert "notifService" not in svc_fields
    
    # Assert Saga command extracted via balanced brace parsing
    assert res["saga_commands"] == ["updateProjectUser"]
    
    # Assert Java hop followed SagasConfig to UpdateProjectUserHandler
    roles = [bh.get("role") for bh in res["backend_handlers"]]
    assert "Saga Configuration" in roles
    assert "Saga Handler" in roles
    handler_entry = next(bh for bh in res["backend_handlers"] if bh.get("role") == "Saga Handler")
    assert handler_entry["class"] == "UpdateProjectUserHandler"
    assert "memberService.updateMember" in handler_entry["invoked_services"]


def test_flow_build_dependency_context_injects_call_chain(tmp_path):
    """Verify that build_dependency_context in Lean Path Flow resolves call chain from subproject root."""
    from ticket_to_code.lean.scope import ScopeContract, ScopeFence
    from ticket_to_code.lean.lean_path import build_dependency_context

    # Setup fixture monorepo with xchange-ui and project-service
    xchange_dir = tmp_path / "xchange-ui" / "src" / "app" / "modules" / "project" / "coordinators" / "delete-coodinator"
    xchange_dir.mkdir(parents=True)
    i18n_dir = tmp_path / "xchange-ui" / "src" / "assets" / "i18n" / "projects"
    i18n_dir.mkdir(parents=True)
    (i18n_dir / "en.json").write_text('{"project.coordinator.delete.half.msg": "will revoke access from all contracts in this project."}', encoding="utf-8")
    
    html_file = xchange_dir / "delete-coodinator.component.html"
    html_file.write_text('<p>{{ "project.coordinator.delete.half.msg" | translate }}</p>', encoding="utf-8")
    
    ts_file = xchange_dir / "delete-coodinator.component.ts"
    ts_file.write_text('''
    export class DeleteCoordinatorComponent {
      constructor(private memberService: MemberService) {}
      deleteCoordinator() {
        this.memberService.updateProjectUsers("p-1", {});
      }
    }
    ''', encoding="utf-8")
    
    service_dir = tmp_path / "xchange-ui" / "src" / "app" / "services"
    service_dir.mkdir(parents=True)
    (service_dir / "member.service.ts").write_text('''
    export class MemberService {
      updateProjectUsers(p: string, m: any) {
        return this.sagaService.doCommand("updateProjectUser", { p, m });
      }
    }
    ''', encoding="utf-8")
    
    java_config_dir = tmp_path / "project-service" / "src" / "main" / "java" / "com" / "cloudforge" / "sagas"
    java_config_dir.mkdir(parents=True)
    (java_config_dir / "SagasConfig.java").write_text('''
    public class SagasConfig {
      private final UpdateProjectUserHandler updateProjectUserHandler;
      public void configure() {
        step.addStepHandler("updateProjectUser", updateProjectUserHandler);
      }
    }
    ''', encoding="utf-8")
    (java_config_dir / "UpdateProjectUserHandler.java").write_text('''
    public class UpdateProjectUserHandler {
      public void process(SagaState state) {
        memberService.updateMember("p-1", member);
      }
    }
    ''', encoding="utf-8")

    # Pass scope pointing to SUBPROJECT root (tmp_path / xchange-ui)
    subproject_root = str(tmp_path / "xchange-ui")
    scope = ScopeContract(
        ticket_id="TICKET-1",
        layer="ui",
        confidence=0.9,
        root=subproject_root,
        allow_globs=("src/**",),
        deny_globs=(),
        anchors=("will revoke access from all contracts in this project", "delete-coodinator"),
        ticket_text="Remove coordinator revokes contracts",
    )
    fence = ScopeFence(scope)
    locked_rel = ["src/app/modules/project/coordinators/delete-coodinator/delete-coodinator.component.ts"]
    
    dep_ctx = build_dependency_context(scope, locked_rel, fence)
    
    # Assert that Flow context contains the candidate fullstack execution chain
    assert "CANDIDATE FULLSTACK EXECUTION CHAIN (HEURISTIC / UNVERIFIED):" in dep_ctx
    assert "updateProjectUser" in dep_ctx
    assert "UpdateProjectUserHandler" in dep_ctx


def test_real_triage_coordinator_ticket_injects_call_chain(tmp_path):
    """Test fullstack call chain injection using real triage on coordinator ticket."""
    from ticket_to_code.lean.scope import triage, ScopeFence
    from ticket_to_code.lean.lean_path import build_dependency_context

    # Setup fixture monorepo
    xchange_dir = tmp_path / "xchange-ui" / "src" / "app" / "modules" / "project" / "coordinators" / "delete-coodinator"
    xchange_dir.mkdir(parents=True)
    i18n_dir = tmp_path / "xchange-ui" / "src" / "assets" / "i18n" / "projects"
    i18n_dir.mkdir(parents=True)
    (i18n_dir / "en.json").write_text('{"project.coordinator.delete.half.msg": "will revoke access from all contracts in this project."}', encoding="utf-8")
    
    html_file = xchange_dir / "delete-coodinator.component.html"
    html_file.write_text('<p>{{ "project.coordinator.delete.half.msg" | translate }}</p>', encoding="utf-8")
    
    ts_file = xchange_dir / "delete-coodinator.component.ts"
    ts_file.write_text('''
    export class DeleteCoordinatorComponent {
      constructor(private memberService: MemberService) {}
      deleteCoordinator() {
        this.memberService.updateProjectUsers("p-1", {});
      }
    }
    ''', encoding="utf-8")
    
    service_dir = tmp_path / "xchange-ui" / "src" / "app" / "services"
    service_dir.mkdir(parents=True)
    (service_dir / "member.service.ts").write_text('''
    export class MemberService {
      updateProjectUsers(p: string, m: any) {
        return this.sagaService.doCommand("updateProjectUser", { p, m });
      }
    }
    ''', encoding="utf-8")
    
    java_config_dir = tmp_path / "project-service" / "src" / "main" / "java" / "com" / "cloudforge" / "sagas"
    java_config_dir.mkdir(parents=True)
    (java_config_dir / "SagasConfig.java").write_text('''
    public class SagasConfig {
      private final UpdateProjectUserHandler updateProjectUserHandler;
      public void configure() {
        step.addStepHandler("updateProjectUser", updateProjectUserHandler);
      }
    }
    ''', encoding="utf-8")
    (java_config_dir / "UpdateProjectUserHandler.java").write_text('''
    public class UpdateProjectUserHandler {
      public void process(SagaState state) {
        memberService.updateMember("p-1", member);
      }
    }
    ''', encoding="utf-8")

    title = "When removing a co-ordinator at a project level, a message dialog appears that the access for that member would be removed from all contracts"
    desc = 'They are not removed from the contract. Navigate to contract: Observe user still a member. Message: "will revoke access from all contracts in this project."'
    
    # Run REAL triage
    scope = triage("CC4E-100", title, desc, str(tmp_path))
    assert not scope.grid_hint, "Coordinator ticket must not be classified as grid ticket"
    
    fence = ScopeFence(scope)
    locked_rel = ["src/app/modules/project/coordinators/delete-coodinator/delete-coodinator.component.ts"]
    
    dep_ctx = build_dependency_context(scope, locked_rel, fence)
    assert "CANDIDATE FULLSTACK EXECUTION CHAIN (HEURISTIC / UNVERIFIED):" in dep_ctx
    assert "updateProjectUser" in dep_ctx
    assert "UpdateProjectUserHandler" in dep_ctx


def test_real_triage_column_reorder_ticket_skips_call_chain(tmp_path):
    """Test that grid/column-reorder tickets skip tracer completely with 0s latency and zero noise."""
    from ticket_to_code.lean.scope import triage, ScopeFence
    from ticket_to_code.lean.lean_path import build_dependency_context

    title = "Members tab grid column reordering"
    desc = "Reorder columns in members grid: Name, Role, Contract List, Status"
    
    # Run REAL triage
    scope = triage("CC4E-200", title, desc, str(tmp_path))
    assert scope.grid_hint is True, "Column reordering ticket must have grid_hint=True"
    
    fence = ScopeFence(scope)
    locked_rel = ["src/app/modules/project/members/members.component.ts"]
    
    dep_ctx = build_dependency_context(scope, locked_rel, fence)
    assert "EXECUTION CHAIN" not in dep_ctx, "Column reordering ticket must not have call chain injected"


def test_tracer_error_does_not_crash_flow(tmp_path, monkeypatch):
    """Test that a tracer exception logs a warning and does not raise NameError or crash the run."""
    from ticket_to_code.lean.scope import ScopeContract, ScopeFence
    from ticket_to_code.lean.lean_path import build_dependency_context
    from ticket_to_code.lean.call_chain_tracer import CallChainTracer

    def mock_trace_fail(self, query):
        raise RuntimeError("Forced tracer failure for resilience test")

    monkeypatch.setattr(CallChainTracer, "trace", mock_trace_fail)

    scope = ScopeContract(
        ticket_id="TICKET-ERR",
        layer="ui",
        confidence=0.9,
        root=str(tmp_path),
        allow_globs=("src/**",),
        deny_globs=(),
        anchors=("will revoke access from all contracts in this project",),
        ticket_text="Remove coordinator revokes contracts dialog message",
    )
    fence = ScopeFence(scope)
    locked_rel = ["src/app/test.ts"]

    # Must complete safely without raising NameError or RuntimeError
    dep_ctx = build_dependency_context(scope, locked_rel, fence)
    assert isinstance(dep_ctx, str)


def test_call_chain_tracer_discovers_architecture_and_sibling_sagas(tmp_path):
    """Test that CallChainTracer identifies actual monorepo services and existing sibling Sagas to prevent hallucinated services."""
    from ticket_to_code.lean.call_chain_tracer import CallChainTracer

    # Create monorepo services
    (tmp_path / "project-service").mkdir()
    (tmp_path / "project-service" / "pom.xml").write_text("<project></project>", encoding="utf-8")
    (tmp_path / "area-service").mkdir()
    (tmp_path / "area-service" / "pom.xml").write_text("<project></project>", encoding="utf-8")
    (tmp_path / "sagas-service").mkdir()
    (tmp_path / "sagas-service" / "pom.xml").write_text("<project></project>", encoding="utf-8")
    (tmp_path / "xchange-ui").mkdir()
    (tmp_path / "xchange-ui" / "package.json").write_text("{}", encoding="utf-8")

    # Add sagas application.yml
    res_dir = tmp_path / "sagas-service" / "src" / "main" / "resources"
    res_dir.mkdir(parents=True)
    (res_dir / "application.yml").write_text('''
sagas:
  commands:
    removeContractMembers:
      steps:
        removeContractMembers:
          stepRoutingKey: coreforbim.projects
        removeContractMembersToArea:
          stepRoutingKey: engg.areas
''', encoding="utf-8")

    tracer = CallChainTracer(str(tmp_path))
    res = tracer.trace("will revoke access from all contracts in this project")

    # Verify discovered microservices
    assert "project-service" in res["microservices"]
    assert "area-service" in res["microservices"]
    assert "sagas-service" in res["microservices"]
    assert "xchange-ui" in res["microservices"]
    assert "contract-service" not in res["microservices"]

    # Verify discovered sibling sagas
    caps = [c["saga_name"] for c in res["related_domain_capabilities"]]
    assert "removeContractMembers" in caps

    # Verify summary trace includes architecture context and sibling saga
    summary_text = "\n".join(res["summary_trace"])
    assert "Discovered services in repository:" in summary_text
    assert "project-service" in summary_text
    assert "removeContractMembers" in summary_text


def test_generic_monorepo_no_cc4e_hardcoding(tmp_path):
    """Verify that a generic monorepo with unrelated services produces zero CC4E false claims."""
    from ticket_to_code.lean.call_chain_tracer import CallChainTracer

    # Monorepo with completely different services
    (tmp_path / "orders-service").mkdir()
    (tmp_path / "orders-service" / "pom.xml").write_text("<project></project>", encoding="utf-8")
    (tmp_path / "billing-service").mkdir()
    (tmp_path / "billing-service" / "package.json").write_text("{}", encoding="utf-8")

    tracer = CallChainTracer(str(tmp_path))
    res = tracer.trace("Order fulfillment payment failure")

    assert "orders-service" in res["microservices"]
    assert "billing-service" in res["microservices"]

    summary_text = "\n".join(res["summary_trace"])
    # Absolutely zero CC4E specific terms
    assert "orders-service" in summary_text
    assert "billing-service" in summary_text
    assert "contract-service" not in summary_text
    assert "project-service" not in summary_text
    assert "area-service" not in summary_text
    assert "CMS" not in summary_text


def test_handler_archetype_and_ts_model_contract(tmp_path):
    """Verify that tracer extracts real handler super_class/process signatures and TS model fields."""
    from ticket_to_code.lean.call_chain_tracer import CallChainTracer

    # Setup UI with model and component
    ui_dir = tmp_path / "xchange-ui" / "src" / "app"
    (ui_dir / "models").mkdir(parents=True)
    (ui_dir / "models" / "user.ts").write_text('''
    export interface AccountUser {
      id: string;
      userId: string;
      otdsUserId: string;
      roleName: string;
    }
    ''', encoding="utf-8")

    (ui_dir / "components").mkdir(parents=True)
    (ui_dir / "components" / "delete-user.component.html").write_text('<button (click)="delete()">removeUser</button>', encoding="utf-8")
    (ui_dir / "components" / "delete-user.component.ts").write_text('''
    import { AccountUser } from '../models/user';
    export class DeleteUserComponent {
      currUser: AccountUser;
      constructor(private userService: UserService) {}
      delete() {
        this.userService.removeUser(this.currUser);
      }
    }
    ''', encoding="utf-8")

    (ui_dir / "services").mkdir(parents=True)
    (ui_dir / "services" / "user.service.ts").write_text('''
    export class UserService {
      removeUser(u: any) {
        return this.sagaService.doCommand("removeUser", { u });
      }
    }
    ''', encoding="utf-8")

    # Setup Java backend handler extending an abstract base class
    java_dir = tmp_path / "user-service" / "src" / "main" / "java" / "com" / "example"
    java_dir.mkdir(parents=True)
    (java_dir / "SagasConfig.java").write_text('''
    public class SagasConfig {
      public void config() {
        step.addStepHandler("removeUser", RemoveUserHandler);
      }
    }
    ''', encoding="utf-8")
    (java_dir / "RemoveUserHandler.java").write_text('''
    public class RemoveUserHandler extends AbstractStepHandler {
      @Autowired
      private UserRepository userRepository;

      @Override
      protected void process(SagaState state) throws Exception {
        UUID uId = getFieldAsUUID("userId", state);
        userRepository.deleteById(uId);
        setFieldForNextStep("deleted", true, state);
      }
    }
    ''', encoding="utf-8")

    tracer = CallChainTracer(str(tmp_path))
    res = tracer.trace("removeUser")

    # Verify model contract extraction
    assert len(res["model_contracts"]) > 0
    mc = res["model_contracts"][0]
    assert mc["model_name"] == "AccountUser"
    assert "otdsUserId" in mc["properties"]
    assert "userId" in mc["properties"]

    # Verify backend handler archetype extraction
    handlers = [h for h in res["backend_handlers"] if h.get("role") == "Saga Handler"]
    assert len(handlers) > 0
    h = handlers[0]
    assert h["super_class"] == "AbstractStepHandler"
    assert "process(SagaState state)" in h["entry_method"]
    assert "getFieldAsUUID" in h["state_idioms"]

    # Verify summary trace includes guidance note and model contract
    summary = "\n".join(res["summary_trace"])
    assert "Frontend Model Contract" in summary
    assert "otdsUserId" in summary
    assert "Backend Handler Archetype" in summary
    assert "AbstractStepHandler" in summary
    assert "ARCHITECTURAL NOTE" in summary


def test_non_cc4e_monorepo_with_repository_methods_and_candidate_label(tmp_path):
    """Verify that a completely non-CC4E monorepo discovers real repository methods and labels candidate chains without false architecture claims."""
    from ticket_to_code.lean.call_chain_tracer import CallChainTracer

    # 1. Setup UI service
    ui_dir = tmp_path / "catalog-ui" / "src" / "app"
    (ui_dir / "items").mkdir(parents=True)
    (ui_dir / "items" / "item.component.html").write_text('<button (click)="removeItem()">Delete Product</button>', encoding="utf-8")
    (ui_dir / "items" / "item.component.ts").write_text('''
    export class ItemComponent {
      constructor(private catalogService: CatalogService) {}
      removeItem() {
        this.catalogService.deleteProduct("SKU-100");
      }
    }
    ''', encoding="utf-8")

    (ui_dir / "services").mkdir(parents=True)
    (ui_dir / "services" / "catalog.service.ts").write_text('''
    export class CatalogService {
      deleteProduct(sku: string) {
        return this.client.delete("/api/products/" + sku);
      }
    }
    ''', encoding="utf-8")

    # 2. Setup backend service with entity and repository
    backend_model_dir = tmp_path / "inventory-service" / "src" / "main" / "java" / "com" / "ecommerce" / "model"
    backend_model_dir.mkdir(parents=True)
    (backend_model_dir / "ProductItem.java").write_text('''
    package com.ecommerce.model;
    import javax.persistence.Entity;
    import javax.persistence.Table;

    @Entity
    @Table(name = "product_items")
    public class ProductItem {
      private String sku;
      private String name;
    }
    ''', encoding="utf-8")

    backend_repo_dir = tmp_path / "inventory-service" / "src" / "main" / "java" / "com" / "ecommerce" / "repository"
    backend_repo_dir.mkdir(parents=True)
    (backend_repo_dir / "ProductItemRepository.java").write_text('''
    package com.ecommerce.repository;
    import com.ecommerce.model.ProductItem;
    import org.springframework.data.jpa.repository.JpaRepository;
    import org.springframework.stereotype.Repository;

    @Repository
    public interface ProductItemRepository extends JpaRepository<ProductItem, String> {
      ProductItem findBySku(String sku);
      void deleteBySku(String sku);
      long countBySku(String sku);
    }
    ''', encoding="utf-8")

    (tmp_path / "catalog-ui" / "package.json").write_text("{}", encoding="utf-8")
    (tmp_path / "inventory-service" / "pom.xml").write_text("<project></project>", encoding="utf-8")

    tracer = CallChainTracer(str(tmp_path))
    res = tracer.trace("Delete Product item")

    # Verify discovered microservices
    assert "catalog-ui" in res["microservices"]
    assert "inventory-service" in res["microservices"]
    assert "project-service" not in res["microservices"]
    assert "area-service" not in res["microservices"]

    # Verify domain entity & extracted repository methods
    assert len(res["domain_entities"]) > 0
    de = res["domain_entities"][0]
    assert de["entity"] == "ProductItem"
    assert de["table"] == "product_items"
    assert de["service"] == "inventory-service"
    assert "findBySku" in de["repo_methods"]
    assert "deleteBySku" in de["repo_methods"]

    # Verify summary trace text
    summary = "\n".join(res["summary_trace"])
    assert "product_items" in summary
    assert "findBySku" in summary
    assert "deleteBySku" in summary
    assert "Only use verified declared repository methods; do NOT invent non-existent query methods." in summary
    assert "project-service" not in summary
    assert "area-service" not in summary


def test_arbitrary_named_yaml_workflow_discovery(tmp_path):
    """Verify that workflows and sibling sagas are discovered from any YAML/JSON file without filename keyword restrictions."""
    from ticket_to_code.lean.call_chain_tracer import CallChainTracer

    # Service with config file NOT named application/saga/workflow (e.g. routing-pipeline.yml)
    svc_dir = tmp_path / "order-orchestrator" / "src" / "main" / "resources"
    svc_dir.mkdir(parents=True)
    (tmp_path / "order-orchestrator" / "pom.xml").write_text("<project></project>", encoding="utf-8")

    (svc_dir / "routing-pipeline.yml").write_text('''
    sagas:
      cancelOrder:
        completionRoutingKey: orders.completed
        steps:
          verifyOrderState:
            stepRoutingKey: orders.state
          releaseHold:
            stepRoutingKey: payments.hold
      refundPayment:
        completionRoutingKey: payments.refunded
        steps:
          processRefund:
            stepRoutingKey: payments.process
    ''', encoding="utf-8")

    # UI triggering cancelOrder
    ui_dir = tmp_path / "orders-ui" / "src" / "app" / "services"
    ui_dir.mkdir(parents=True)
    (tmp_path / "orders-ui" / "package.json").write_text("{}", encoding="utf-8")
    (ui_dir / "orders.service.ts").write_text('''
    export class OrdersService {
      cancelOrder(id: string) {
        return this.client.doCommand("cancelOrder", { id });
      }
    }
    ''', encoding="utf-8")

    tracer = CallChainTracer(str(tmp_path))
    # Directly test workflow definitions extraction
    wfs = tracer._find_workflow_definitions(["cancelOrder"])
    assert len(wfs) == 1
    assert wfs[0]["saga_name"] == "cancelOrder"
    assert len(wfs[0]["steps"]) == 2
    assert wfs[0]["steps"][0]["step"] == "verifyOrderState"

    # Test sibling capability discovery from the arbitrary-named YAML
    caps = tracer._find_related_domain_capabilities("Cancel Order and refund payment", {"payment", "order"})
    assert any(c["saga_name"] == "refundPayment" for c in caps)


def test_call_chain_tracer_resolves_interface_to_concrete_implementation(tmp_path):
    """Verifies that CallChainTracer resolves Java service interfaces to concrete implementations
    and extracts the targeted method body."""
    from ticket_to_code.lean.call_chain_tracer import CallChainTracer

    svc_dir = tmp_path / "src" / "main" / "java" / "com" / "example" / "services"
    svc_dir.mkdir(parents=True)
    (svc_dir / "PaymentService.java").write_text("public interface PaymentService { void processRefund(String id); }", encoding="utf-8")
    (svc_dir / "PaymentServiceImpl.java").write_text("""
    public class PaymentServiceImpl implements PaymentService {
        public void processRefund(String id) {
            recordAudit(id);
            gateway.execute(id);
        }
    }
    """, encoding="utf-8")

    tracer = CallChainTracer(str(tmp_path))
    impls = tracer._get_interface_implementations("PaymentService")
    assert len(impls) >= 1
    assert any("PaymentServiceImpl" in item[1] for item in impls)

    # Verify method body extraction
    body = tracer._extract_method_body(
        (svc_dir / "PaymentServiceImpl.java").read_text(encoding="utf-8"),
        "processRefund"
    )
    assert body is not None
    assert "recordAudit(id);" in body





