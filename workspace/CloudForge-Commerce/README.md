# CloudForge Commerce (Polyglot Showcase Monorepo)

An open-source reference e-commerce and settlement platform used for autonomous SWE agent evaluation and benchmarking.

## Architecture

* **Backend (`order-service`)**:
  * Spring Boot 3.2.3, Java 17, Spring Data JPA, H2 In-Memory DB, JUnit 5.
  * Manages order lifecycle and synchronizes payment holds with `PaymentLedgerService`.
* **Frontend (`store-ui`)**:
  * React 18, TypeScript, Vite.
  * Displays real-time order status, payment reservation state, and cancellation workflows.

## Known Issue: CF-204

* **Symptom**: When an order is cancelled, `OrderService.cancelOrder()` transitions order status to `CANCELLED`, but fails to release the linked `PaymentReservation` in `PaymentLedgerService`.
* **Impact**: Orphaned payment holds prevent customers from reusing credit lines.
* **Reproduction**:
  ```bash
  cd order-service
  mvn test -Dtest=OrderServiceTest#testCancelOrderReleasesPaymentReservationHold
  ```
