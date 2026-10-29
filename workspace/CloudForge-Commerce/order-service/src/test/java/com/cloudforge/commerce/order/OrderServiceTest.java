package com.cloudforge.commerce.order;

import com.cloudforge.commerce.order.model.Order;
import com.cloudforge.commerce.order.model.OrderStatus;
import com.cloudforge.commerce.order.model.PaymentReservation;
import com.cloudforge.commerce.order.model.ReservationStatus;
import com.cloudforge.commerce.order.repository.OrderRepository;
import com.cloudforge.commerce.order.repository.PaymentReservationRepository;
import com.cloudforge.commerce.order.service.OrderService;
import com.cloudforge.commerce.order.service.PaymentLedgerService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.math.BigDecimal;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest(classes = OrderServiceApplication.class)
public class OrderServiceTest {

    @Autowired
    private OrderService orderService;

    @Autowired
    private PaymentLedgerService paymentLedgerService;

    @Autowired
    private OrderRepository orderRepository;

    @Autowired
    private PaymentReservationRepository reservationRepository;

    @BeforeEach
    void setUp() {
        orderRepository.deleteAll();
        reservationRepository.deleteAll();
    }

    @Test
    @DisplayName("Placing an order successfully creates an active payment reservation hold")
    void testPlaceOrderCreatesActiveReservation() {
        Order order = orderService.placeOrder("CUST-1001", new BigDecimal("149.99"));
        assertNotNull(order.getId());
        assertEquals(OrderStatus.PENDING, order.getStatus());
        assertNotNull(order.getReservationId());

        ReservationStatus resStatus = paymentLedgerService.getReservationStatus(order.getReservationId());
        assertEquals(ReservationStatus.ACTIVE, resStatus);
    }

    @Test
    @DisplayName("CF-204: Cancelling an order MUST automatically release the payment reservation hold")
    void testCancelOrderReleasesPaymentReservationHold() {
        // Given an order with an active reservation hold
        Order order = orderService.placeOrder("CUST-1002", new BigDecimal("250.00"));
        Long resId = order.getReservationId();
        assertEquals(ReservationStatus.ACTIVE, paymentLedgerService.getReservationStatus(resId));

        // When order is cancelled
        Order cancelled = orderService.cancelOrder(order.getId(), "Customer changed mind");
        assertEquals(OrderStatus.CANCELLED, cancelled.getStatus());

        // Then the reservation hold MUST be released (CF-204 assertion)
        ReservationStatus finalResStatus = paymentLedgerService.getReservationStatus(resId);
        assertEquals(ReservationStatus.RELEASED, finalResStatus, 
            "CF-204 Violation: Cancelled order left orphaned ACTIVE reservation hold in PaymentLedgerService!");
    }
}
