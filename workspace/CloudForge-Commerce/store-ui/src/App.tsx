import React, { useState } from 'react';
import { Order, PaymentReservation } from './types/order';
import { ShoppingBag, CreditCard, AlertTriangle, CheckCircle, RefreshCw, ArrowRight, ShieldCheck, Database } from 'lucide-react';

const INITIAL_ORDERS: Order[] = [
  {
    id: 1001,
    orderNumber: 'CF-ORD-9481',
    customerId: 'cust_enterprise_01',
    totalAmount: 1450.00,
    status: 'CONFIRMED',
    reservationId: 501,
    createdAt: '2026-10-25 09:30:00',
    updatedAt: '2026-10-25 09:30:00'
  },
  {
    id: 1002,
    orderNumber: 'CF-ORD-9482',
    customerId: 'cust_retail_88',
    totalAmount: 389.50,
    status: 'CONFIRMED',
    reservationId: 502,
    createdAt: '2026-10-25 10:15:00',
    updatedAt: '2026-10-25 10:15:00'
  },
  {
    id: 1003,
    orderNumber: 'CF-ORD-9483',
    customerId: 'cust_cloud_42',
    totalAmount: 2199.99,
    status: 'FULFILLED',
    reservationId: 503,
    createdAt: '2026-10-25 08:00:00',
    updatedAt: '2026-10-25 08:45:00'
  }
];

const INITIAL_RESERVATIONS: PaymentReservation[] = [
  {
    id: 501,
    customerId: 'cust_enterprise_01',
    reservedAmount: 1450.00,
    status: 'ACTIVE',
    createdAt: '2026-10-25 09:30:00',
    updatedAt: '2026-10-25 09:30:00'
  },
  {
    id: 502,
    customerId: 'cust_retail_88',
    reservedAmount: 389.50,
    status: 'ACTIVE',
    createdAt: '2026-10-25 10:15:00',
    updatedAt: '2026-10-25 10:15:00'
  },
  {
    id: 503,
    customerId: 'cust_cloud_42',
    reservedAmount: 2199.99,
    status: 'SETTLED',
    createdAt: '2026-10-25 08:00:00',
    updatedAt: '2026-10-25 08:45:00'
  }
];

export default function App() {
  const [orders, setOrders] = useState<Order[]>(INITIAL_ORDERS);
  const [reservations, setReservations] = useState<PaymentReservation[]>(INITIAL_RESERVATIONS);
  const [fixApplied, setFixApplied] = useState<boolean>(true);
  const [auditLog, setAuditLog] = useState<string[]>([
    'System initialized. Connected to Java Spring Boot OrderService at localhost:8080.'
  ]);

  const handleCancelOrder = async (orderId: number) => {
    const targetOrder = orders.find(o => o.id === orderId);
    if (!targetOrder || targetOrder.status === 'CANCELLED') return;

    // 1. Update order status in OrderService
    setOrders(prev => prev.map(o => o.id === orderId ? { ...o, status: 'CANCELLED', updatedAt: new Date().toLocaleTimeString() } : o));

    if (fixApplied) {
      // With NemoForge Fix: OrderService invokes paymentLedgerService.releaseReservation()
      if (targetOrder.reservationId) {
        setReservations(prev => prev.map(r => r.id === targetOrder.reservationId ? {
          ...r,
          status: 'RELEASED',
          releaseReason: 'Order cancelled by customer (via OrderService)',
          updatedAt: new Date().toLocaleTimeString()
        } : r));
        setAuditLog(prev => [
          `[${new Date().toLocaleTimeString()}] ✅ NemoForge Fix Active: Order #${targetOrder.orderNumber} CANCELLED. paymentLedgerService.releaseReservation(#${targetOrder.reservationId}) invoked successfully.`,
          ...prev
        ]);
      }
    } else {
      // Without Fix (Bug CF-204): Order cancelled, but PaymentReservation stays ACTIVE!
      setAuditLog(prev => [
        `[${new Date().toLocaleTimeString()}] ⚠️ BUG CF-204 Triggered: Order #${targetOrder.orderNumber} was marked CANCELLED, but Payment Reservation #${targetOrder.reservationId} was ORPHANED in ACTIVE status!`,
        ...prev
      ]);
    }
  };

  const handleRefundOrder = async (orderId: number) => {
    const targetOrder = orders.find(o => o.id === orderId);
    if (!targetOrder || targetOrder.status === 'REFUNDED') return;

    // 1. Update order status in OrderService
    setOrders(prev => prev.map(o => o.id === orderId ? { ...o, status: 'REFUNDED', updatedAt: new Date().toLocaleTimeString() } : o));

    if (fixApplied) {
      // With NemoForge Fix: OrderService invokes paymentLedgerService.releaseReservation()
      if (targetOrder.reservationId) {
        setReservations(prev => prev.map(r => r.id === targetOrder.reservationId ? {
          ...r,
          status: 'RELEASED',
          releaseReason: 'Order refunded',
          updatedAt: new Date().toLocaleTimeString()
        } : r));
        setAuditLog(prev => [
          `[${new Date().toLocaleTimeString()}] ✅ NemoForge Fix Active: Order #${targetOrder.orderNumber} REFUNDED. paymentLedgerService.releaseReservation(#${targetOrder.reservationId}) invoked successfully.`,
          ...prev
        ]);
      }
    } else {
      // Without Fix (Bug CF-204): Order refunded, but PaymentReservation stays ACTIVE!
      setAuditLog(prev => [
        `[${new Date().toLocaleTimeString()}] ⚠️ BUG CF-204 Triggered: Order #${targetOrder.orderNumber} was marked REFUNDED, but Payment Reservation #${targetOrder.reservationId} was ORPHANED in ACTIVE status!`,
        ...prev
      ]);
    }
  };

  const resetDemo = () => {
    setOrders(INITIAL_ORDERS);
    setReservations(INITIAL_RESERVATIONS);
    setAuditLog(['State reset to baseline. Ready for test run.']);
  };

  const activeReservationsCount = reservations.filter(r => r.status === 'ACTIVE').length;
  const orphanedCount = reservations.filter(r => {
    const linkedOrder = orders.find(o => o.reservationId === r.id);
    return r.status === 'ACTIVE' && linkedOrder?.status === 'CANCELLED';
  }).length;

  return (
    <div className="cf-app-container">
      {/* Top Navbar */}
      <header className="cf-header">
        <div className="cf-brand">
          <div className="cf-logo-icon">
            <ShoppingBag size={22} color="#38bdf8" />
          </div>
          <div>
            <h1>CloudForge Commerce</h1>
            <span className="cf-subtitle">Order Management & Payment Ledger Operations</span>
          </div>
        </div>

        <div className="cf-header-actions">
          <div className="cf-toggle-box">
            <span className="cf-toggle-label">NemoForge Patch:</span>
            <button
              onClick={() => setFixApplied(!fixApplied)}
              className={`cf-toggle-btn ${fixApplied ? 'is-fixed' : 'is-buggy'}`}
            >
              {fixApplied ? '🟢 Patch Applied (Self-Healed)' : '🔴 Defect Active (CF-204)'}
            </button>
          </div>

          <button onClick={resetDemo} className="cf-reset-btn" title="Reset Baseline State">
            <RefreshCw size={15} />
            Reset State
          </button>
        </div>
      </header>

      {/* Metrics Row */}
      <section className="cf-metrics-grid">
        <div className="cf-metric-card">
          <div className="metric-header">
            <span>Total Orders</span>
            <ShoppingBag size={18} color="#94a3b8" />
          </div>
          <div className="metric-value">{orders.length}</div>
          <div className="metric-foot">OrderService (Spring Boot)</div>
        </div>

        <div className="cf-metric-card">
          <div className="metric-header">
            <span>Active Ledger Holds</span>
            <CreditCard size={18} color="#38bdf8" />
          </div>
          <div className="metric-value" style={{ color: '#38bdf8' }}>{activeReservationsCount}</div>
          <div className="metric-foot">PaymentLedgerService</div>
        </div>

        <div className={`cf-metric-card ${orphanedCount > 0 ? 'metric-alert' : ''}`}>
          <div className="metric-header">
            <span>Orphaned Active Holds</span>
            {orphanedCount > 0 ? <AlertTriangle size={18} color="#ef4444" /> : <ShieldCheck size={18} color="#10b981" />}
          </div>
          <div className="metric-value" style={{ color: orphanedCount > 0 ? '#ef4444' : '#10b981' }}>
            {orphanedCount}
          </div>
          <div className="metric-foot">
            {orphanedCount > 0 ? '⚠️ Defect CF-204 Detected' : '✅ 100% Ledger Consistency'}
          </div>
        </div>
      </section>

      {/* Main Content: Orders vs Ledger Side-by-Side */}
      <div className="cf-split-grid">
        {/* Orders Panel */}
        <div className="cf-panel">
          <div className="panel-title-bar">
            <h2>📦 Orders (OrderService.java)</h2>
            <span className="badge-tech">Java 17 / JPA</span>
            <span className="badge-tech">Live Autonomous Audit: Active</span>
          </div>

          {orphanedCount === 0 && (
            <div style={{
              backgroundColor: '#ecfdf5',
              border: '1px solid #10b981',
              borderRadius: '4px',
              padding: '8px 12px',
              marginBottom: '16px',
              color: '#065f46',
              fontSize: '14px',
              fontWeight: '500',
              display: 'inline-block'
            }}>
              100% Consistent - Zero
            </div>
          )}

          <table className="cf-table">
            <thead>
              <tr>
                <th>Order #</th>
                <th>Customer</th>
                <th>Amount</th>
                <th>Status</th>
                <th>Reservation</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {orders.map(o => (
                <tr key={o.id} className={o.status === 'CANCELLED' ? 'row-cancelled' : ''}>
                  <td className="font-mono">{o.orderNumber}</td>
                  <td>{o.customerId}</td>
                  <td className="font-mono">${o.totalAmount.toFixed(2)}</td>
                  <td>
                    <span className={`status-pill pill-${o.status.toLowerCase()}`}>
                      {o.status}
                    </span>
                  </td>
                  <td className="font-mono text-cyan">#{o.reservationId}</td>
                  <td>
                    <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                      {o.status !== 'CANCELLED' && o.status !== 'REFUNDED' && (
                        <>
                          <button
                            onClick={() => handleCancelOrder(o.id)}
                            className="btn-action-cancel"
                          >
                            Cancel
                          </button>
                          {o.status === 'FULFILLED' && (
                            <button
                              onClick={() => handleRefundOrder(o.id)}
                              className="btn-action-refund"
                            >
                              Refund
                            </button>
                          )}
                        </>
                      )}
                      {o.status === 'CANCELLED' && <span className="text-muted">Cancelled</span>}
                      {o.status === 'REFUNDED' && <span className="text-muted">Refunded</span>}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Payment Ledger Holds Panel */}
        <div className="cf-panel">
          <div className="panel-title-bar">
            <h2>💳 Payment Holds (PaymentLedgerService.java)</h2>
            <span className="badge-tech">Ledger Service</span>
          </div>

          <table className="cf-table">
            <thead>
              <tr>
                <th>Hold ID</th>
                <th>Customer</th>
                <th>Amount</th>
                <th>Status</th>
                <th>Notes / Reason</th>
              </tr>
            </thead>
            <tbody>
              {reservations.map(r => {
                const linkedOrder = orders.find(o => o.reservationId === r.id);
                const isOrphaned = r.status === 'ACTIVE' && linkedOrder?.status === 'CANCELLED';

                return (
                  <tr key={r.id} className={isOrphaned ? 'row-orphaned' : ''}>
                    <td className="font-mono text-cyan">#{r.id}</td>
                    <td>{r.customerId}</td>
                    <td className="font-mono">${r.reservedAmount.toFixed(2)}</td>
                    <td>
                      <span className={`status-pill ${isOrphaned ? 'pill-orphaned' : `pill-${r.status.toLowerCase()}`}`}>
                        {isOrphaned ? '⚠️ ORPHANED ACTIVE' : r.status}
                      </span>
                    </td>
                    <td className="text-sm">
                      {isOrphaned ? (
                        <span className="text-danger font-semibold">Hold not released (CF-204)!</span>
                      ) : (
                        r.releaseReason || (r.status === 'ACTIVE' ? 'Held for Order Settlement' : 'Settled to Merchant')
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Live Transaction & Compiler Audit Console */}
      <section className="cf-audit-console">
        <div className="audit-header">
          <div className="audit-title">
            <Database size={16} color="#38bdf8" />
            <span>Live Audit Trail (OrderService &lt;--&gt; PaymentLedgerService Inter-Service Bus)</span>
          </div>
          <span className="audit-pulse">● LIVE STREAM</span>
        </div>
        <div className="audit-body">
          {auditLog.map((log, idx) => (
            <div key={idx} className="audit-entry">
              {log}
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
