import subprocess
from pathlib import Path

# 1. Reset baseline
subprocess.run(["python", "nemoforge/scripts/reset_baseline.py"], check=True)

# 2. Add REFUNDED to OrderStatus.java
status_path = Path("workspace/CloudForge-Commerce/order-service/src/main/java/com/cloudforge/commerce/order/model/OrderStatus.java")
status_text = status_path.read_text(encoding="utf-8")
status_text = status_text.replace("CANCELLED\n}", "CANCELLED,\n    REFUNDED\n}")
status_path.write_text(status_text, encoding="utf-8")

# 3. Add refundOrder to OrderService.java before the final closing brace
service_path = Path("workspace/CloudForge-Commerce/order-service/src/main/java/com/cloudforge/commerce/order/service/OrderService.java")
service_text = service_path.read_text(encoding="utf-8")
refund_method = """
    @Transactional
    public Order refundOrder(Long orderId) {
        log.info("Refunding order id: {}", orderId);
        Order order = orderRepository.findById(orderId)
                .orElseThrow(() -> new IllegalArgumentException("Order not found: " + orderId));

        if (order.getStatus() != OrderStatus.FULFILLED) {
            log.warn("Order {} is not fulfilled and cannot be refunded. Current status: {}", orderId, order.getStatus());
            throw new IllegalStateException("Only fulfilled orders can be refunded");
        }

        order.setStatus(OrderStatus.REFUNDED);
        paymentLedgerService.releaseReservation(order.getReservationId(), "Order refunded");
        return orderRepository.save(order);
    }
"""
last_b = service_text.rfind("}")
service_text = service_text[:last_b] + refund_method + service_text[last_b:]
service_path.write_text(service_text, encoding="utf-8")

# 4. Add refund endpoint to OrderController.java
ctrl_path = Path("workspace/CloudForge-Commerce/order-service/src/main/java/com/cloudforge/commerce/order/controller/OrderController.java")
ctrl_text = ctrl_path.read_text(encoding="utf-8")
endpoint = """
    @PostMapping("/{orderId}/refund")
    public ResponseEntity<Order> refundOrder(@PathVariable Long orderId) {
        Order refunded = orderService.refundOrder(orderId);
        return ResponseEntity.ok(refunded);
    }
"""
last_cb = ctrl_text.rfind("}")
ctrl_text = ctrl_text[:last_cb] + endpoint + ctrl_text[last_cb:]
ctrl_path.write_text(ctrl_text, encoding="utf-8")

# 5. Compile with maven
cmd = "cd workspace/CloudForge-Commerce/order-service && mvn clean test-compile"
res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print("MAVEN RETURN CODE:", res.returncode)
if "BUILD SUCCESS" in res.stdout:
    print("BUILD SUCCESS VERIFIED on Java order-service!")
else:
    print("STDOUT:", res.stdout[-1500:])
    print("STDERR:", res.stderr[-1500:])

