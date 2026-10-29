package com.cloudforge.commerce.order.controller;

import com.cloudforge.commerce.order.model.Order;
import com.cloudforge.commerce.order.service.OrderService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.math.BigDecimal;
import java.util.Map;

@RestController
@RequestMapping("/api/orders")
@RequiredArgsConstructor
public class OrderController {

    private final OrderService orderService;

    @PostMapping
    public ResponseEntity<Order> createOrder(@RequestBody Map<String, Object> payload) {
        String customerId = (String) payload.get("customerId");
        BigDecimal amount = new BigDecimal(payload.get("amount").toString());
        Order order = orderService.placeOrder(customerId, amount);
        return ResponseEntity.ok(order);
    }

    @PostMapping("/{orderId}/cancel")
    public ResponseEntity<Order> cancelOrder(
            @PathVariable Long orderId,
            @RequestBody(required = false) Map<String, String> payload) {
        String reason = payload != null && payload.containsKey("reason") 
                ? payload.get("reason") 
                : "Customer requested cancellation";
        Order cancelled = orderService.cancelOrder(orderId, reason);
        return ResponseEntity.ok(cancelled);
    }

    @PostMapping("/{orderId}/refund")
    public ResponseEntity<Order> refundOrder(@PathVariable Long orderId) {
        Order refunded = orderService.refundOrder(orderId);
        return ResponseEntity.ok(refunded);
    }
}
