import React, { useState } from 'react';
import { Order, OrderStatus } from '../types/order';

interface OrderManagementProps {
  orders: Order[];
  onCancelOrder: (orderId: number, reason: string) => Promise<void>;
  onRefundOrder: (orderId: number, reason: string) => Promise<void>;
}

export const OrderManagement: React.FC<OrderManagementProps> = ({ orders, onCancelOrder, onRefundOrder }) => {
  const [cancellingId, setCancellingId] = useState<number | null>(null);
  const [refundingId, setRefundingId] = useState<number | null>(null);

  const handleCancel = async (orderId: number) => {
    setCancellingId(orderId);
    try {
      await onCancelOrder(orderId, 'User requested cancellation from UI');
    } finally {
      setCancellingId(null);
    }
  };

  const handleRefund = async (orderId: number) => {
    setRefundingId(orderId);
    try {
      await onRefundOrder(orderId, 'User requested refund from UI');
    } finally {
      setRefundingId(null);
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
                   {o.status !== 'CANCELLED' && o.status !== 'REFUNDED' && (
                     <>
                       {o.status === 'FULFILLED' && (
                         <button
                           onClick={() => handleRefund(o.id)}
                           disabled={refundingId === o.id}
                           className="btn-refund"
                         >
                           {refundingId === o.id ? 'Refunding...' : 'Refund Order'}
                         </button>
                       )}
                       {o.status !== 'FULFILLED' && (
                         <button
                           onClick={() => handleCancel(o.id)}
                           disabled={cancellingId === o.id}
                           className="btn-cancel"
                         >
                           {cancellingId === o.id ? 'Cancelling...' : 'Cancel Order'}
                         </button>
                       )}
                     </>
                   )}
                 </td>            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
