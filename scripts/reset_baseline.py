"""Reset the CloudForge-Commerce demo workspace to its pre-refund baseline.

Paths are resolved relative to this file so the script works from any cwd.
"""
from pathlib import Path
import re

NEMOFORGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = NEMOFORGE_ROOT.parent / "workspace" / "CloudForge-Commerce"
if not WORKSPACE.exists():
    WORKSPACE = NEMOFORGE_ROOT / "workspace" / "CloudForge-Commerce"

print(f"Target workspace: {WORKSPACE}")

# 1. Reset OrderStatus.java
p_os = WORKSPACE / "order-service" / "src" / "main" / "java" / "com" / "cloudforge" / "commerce" / "order" / "model" / "OrderStatus.java"
p_os.write_text("""package com.cloudforge.commerce.order.model;

public enum OrderStatus {
    PENDING,
    CONFIRMED,
    FULFILLED,
    CANCELLED
}
""", encoding='utf-8')
print("  [+] Reset OrderStatus.java")

# 2. Reset OrderService.java
p_svc = WORKSPACE / "order-service" / "src" / "main" / "java" / "com" / "cloudforge" / "commerce" / "order" / "service" / "OrderService.java"
p_svc.write_text("""package com.cloudforge.commerce.order.service;

import com.cloudforge.commerce.order.model.Order;
import com.cloudforge.commerce.order.model.OrderStatus;
import com.cloudforge.commerce.order.model.PaymentReservation;
import com.cloudforge.commerce.order.repository.OrderRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.util.UUID;

@Service
@Slf4j
@RequiredArgsConstructor
public class OrderService {

    private final OrderRepository orderRepository;
    private final PaymentLedgerService paymentLedgerService;

    @Transactional
    public Order placeOrder(String customerId, BigDecimal amount) {
        log.info("Placing order for customer: {}, amount: {}", customerId, amount);
        PaymentReservation reservation = paymentLedgerService.createReservation(customerId, amount);

        Order order = Order.builder()
                .orderNumber("ORD-" + UUID.randomUUID().toString().substring(0, 8).toUpperCase())
                .customerId(customerId)
                .totalAmount(amount)
                .status(OrderStatus.PENDING)
                .reservationId(reservation.getId())
                .build();

        return orderRepository.save(order);
    }

    @Transactional
    public Order cancelOrder(Long orderId, String reason) {
        log.info("Cancelling order id: {}, reason: {}", orderId, reason);
        Order order = orderRepository.findById(orderId)
                .orElseThrow(() -> new IllegalArgumentException("Order not found: " + orderId));

        if (order.getStatus() == OrderStatus.CANCELLED) {
            log.warn("Order {} is already cancelled", orderId);
            return order;
        }

        order.setStatus(OrderStatus.CANCELLED);
        paymentLedgerService.releaseReservation(order.getReservationId(), "Order cancelled");
        return orderRepository.save(order);
    }

    @Transactional
    public Order fulfillOrder(Long orderId) {
        log.info("Fulfilling order id: {}", orderId);
        Order order = orderRepository.findById(orderId)
                .orElseThrow(() -> new IllegalArgumentException("Order not found: " + orderId));

        if (order.getStatus() == OrderStatus.FULFILLED) {
            log.warn("Order {} is already fulfilled", orderId);
            return order;
        }

        order.setStatus(OrderStatus.FULFILLED);
        paymentLedgerService.releaseReservation(order.getReservationId(), "Order fulfilled");
        return orderRepository.save(order);
    }
}
""", encoding='utf-8')
print("  [+] Reset OrderService.java")

# 3. Reset OrderController.java
p_ctrl = WORKSPACE / "order-service" / "src" / "main" / "java" / "com" / "cloudforge" / "commerce" / "order" / "controller" / "OrderController.java"
p_ctrl.write_text("""package com.cloudforge.commerce.order.controller;

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
}
""", encoding='utf-8')
print("  [+] Reset OrderController.java")

# 4. Reset order.ts
p_ots = WORKSPACE / "store-ui" / "src" / "types" / "order.ts"
p_ots.write_text("""export type OrderStatus = 'PENDING' | 'CONFIRMED' | 'FULFILLED' | 'CANCELLED';

export type ReservationStatus = 'HOLD_PENDING' | 'ACTIVE' | 'RELEASED' | 'SETTLED';

export interface Order {
  id: number;
  orderNumber: string;
  customerId: string;
  totalAmount: number;
  status: OrderStatus;
  reservationId?: number;
  createdAt: string;
  updatedAt: string;
}

export interface PaymentReservation {
  id: number;
  customerId: string;
  reservedAmount: number;
  status: ReservationStatus;
  releaseReason?: string;
  createdAt: string;
  updatedAt: string;
}
""", encoding='utf-8')
print("  [+] Reset order.ts")

# 5. Reset OrderManagement.tsx
p_om = WORKSPACE / "store-ui" / "src" / "components" / "OrderManagement.tsx"
p_om.write_text("""import React, { useState } from 'react';
import { Order, OrderStatus } from '../types/order';

interface OrderManagementProps {
  orders: Order[];
  onCancelOrder: (orderId: number, reason: string) => Promise<void>;
}

export const OrderManagement: React.FC<OrderManagementProps> = ({ orders, onCancelOrder }) => {
  const [cancellingId, setCancellingId] = useState<number | null>(null);

  const handleCancel = async (orderId: number) => {
    setCancellingId(orderId);
    try {
      await onCancelOrder(orderId, 'User requested cancellation from UI');
    } finally {
      setCancellingId(null);
    }
  };

  return (
    <div className="order-management-container">
      <h2>Order & Settlement Dashboard</h2>
      <table className="orders-table">
        <thead>
          <tr>
            <th>Order #</th>
            <th>Customer</th>
            <th>Amount</th>
            <th>Status</th>
            <th>Reservation ID</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {orders.map((o) => (
            <tr key={o.id}>
              <td>{o.orderNumber}</td>
              <td>{o.customerId}</td>
              <td>${o.totalAmount.toFixed(2)}</td>
              <td>
                <span className={`status-badge status-${o.status.toLowerCase()}`}>
                  {o.status}
                </span>
              </td>
              <td>{o.reservationId ? `#${o.reservationId}` : 'None'}</td>
              <td>
                {o.status !== 'CANCELLED' && (
                  <button
                    onClick={() => handleCancel(o.id)}
                    disabled={cancellingId === o.id}
                    className="btn-cancel"
                  >
                    {cancellingId === o.id ? 'Cancelling...' : 'Cancel Order'}
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
""", encoding='utf-8')
print("  [+] Reset OrderManagement.tsx")

# Remove any accidental _new files
for rogue in WORKSPACE.glob("**/*_new.*"):
    try:
        rogue.unlink()
        print(f"  [-] Removed rogue file: {rogue.name}")
    except Exception:
        pass

print("Baseline reset completed successfully!")
