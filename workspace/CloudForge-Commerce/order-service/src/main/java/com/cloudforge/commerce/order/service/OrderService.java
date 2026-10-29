package com.cloudforge.commerce.order.service;

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

    @Transactional
    public Order refundOrder(Long orderId) {
        log.info("Refunding order id: {}", orderId);
        Order order = orderRepository.findById(orderId)
                .orElseThrow(() -> new IllegalArgumentException("Order not found: " + orderId));

        if (order.getStatus() == OrderStatus.REFUNDED) {
            log.warn("Order {} is already refunded", orderId);
            return order;
        }

        order.setStatus(OrderStatus.REFUNDED);
        paymentLedgerService.releaseReservation(order.getReservationId(), "Order refunded");
        return orderRepository.save(order);
    }
}
