import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from .config import config

class OrderTool:
    def __init__(self, orders_path: str):
        self.orders_path = orders_path
        self.orders = self._load_orders()
        
    def _load_orders(self) -> Dict[str, Dict]:
        """Load orders from JSON file"""
        try:
            with open(self.orders_path, 'r') as f:
                orders_data = json.load(f)
                
            # Index by order_id
            orders_by_id = {}
            orders_list = orders_data.get('orders', []) if isinstance(orders_data, dict) else orders_data
            for order in orders_list:
                order_id = order.get('order_id')
                if order_id:
                    # Store only non-sensitive fields
                    orders_by_id[order_id] = self._sanitize_order(order)
                    
            return orders_by_id
        except Exception as e:
            print(f"Error loading orders: {e}")
            return {}
    
    def _sanitize_order(self, order: Dict) -> Dict:
        """Remove sensitive fields from order data"""
        sanitized = {}
        
        # Safe fields to expose
        safe_fields = [
            'order_id', 'status', 'items', 'total_amount',
            'created_at', 'updated_at', 'shipping_method'
        ]
        
        # Exclude specific keys that are handled separately or sensitive
        exclude_keys = ['estimated_delivery', 'delivery_estimate', 'customer', 'internal']
        
        for key, value in order.items():
            if key in safe_fields:
                sanitized[key] = value
            elif key not in config.SENSITIVE_FIELDS and key not in exclude_keys:
                # If not explicitly sensitive but not in safe list, include if simple type
                if isinstance(value, (str, int, float, bool, list)) and not isinstance(value, dict):
                    sanitized[key] = value
                    
        # Add friendly status
        if 'status' in order:
            sanitized['friendly_status'] = self._get_friendly_status(order['status'])
            
        # Add delivery estimate if available and order is active
        if order.get('status') not in ['cancelled', 'returned']:
            if 'estimated_delivery' in order:
                sanitized['estimated_delivery'] = order['estimated_delivery']
            elif 'delivery_estimate' in order:
                sanitized['estimated_delivery'] = order['delivery_estimate']
                
        return sanitized
    
    def _get_friendly_status(self, status: str) -> str:
        """Convert status to friendly description"""
        status_map = {
            'pending': "We're preparing your order",
            'processing': "Your order is being processed",
            'shipped': "Your order has been shipped",
            'delivered': "Your order has been delivered",
            'cancelled': "Your order has been cancelled",
            'returned': "Your order has been returned"
        }
        return status_map.get(status.lower(), status)
    
    def lookup_order(self, order_id: str) -> Dict[str, Any]:
        """Look up an order by ID"""
        if not order_id:
            return {
                'error': 'Order ID is required',
                'message': 'Please provide an order ID to look up'
            }
            
        # Normalize input
        normalized_id = order_id.strip().upper()
        
        # Check if order exists
        if normalized_id not in self.orders:
            # Check if it might be a malformed ID (look for similar)
            similar = self._find_similar_orders(normalized_id)
            if similar:
                return {
                    'error': 'Order not found',
                    'message': f'Order {normalized_id} not found. Did you mean: {", ".join(similar[:3])}?',
                    'suggestions': similar[:3]
                }
            else:
                return {
                    'error': 'Order not found',
                    'message': f'Order {normalized_id} was not found. Please check the order ID and try again.'
                }
                
        order = self.orders[normalized_id]
        
        # Check if order is cancelled or returned - don't show delivery estimates
        if order.get('status') in ['cancelled', 'returned']:
            if 'delivery_estimate' in order:
                del order['delivery_estimate']
                
        return {
            'found': True,
            'order': order,
            'message': f"Order {normalized_id} found. Status: {order.get('friendly_status')}"
        }
    
    def _find_similar_orders(self, order_id: str) -> List[str]:
        """Find similar order IDs (fuzzy matching)"""
        similar = []
        for existing_id in self.orders.keys():
            # Simple similarity check
            if order_id in existing_id or existing_id in order_id:
                similar.append(existing_id)
            elif len(order_id) >= 4 and len(existing_id) >= 4:
                # Check if first 4 chars match (common prefix)
                if order_id[:4].upper() == existing_id[:4].upper():
                    similar.append(existing_id)
        return similar
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get just the status of an order"""
        result = self.lookup_order(order_id)
        if result.get('found'):
            return {
                'order_id': order_id,
                'status': result['order'].get('status'),
                'friendly_status': result['order'].get('friendly_status'),
                'message': result['message']
            }
        else:
            return result
    
    def get_order_summary(self, order_id: str) -> Dict[str, Any]:
        """Get a summary of the order (for display)"""
        result = self.lookup_order(order_id)
        if result.get('found'):
            order = result['order']
            summary = {
                'order_id': order.get('order_id'),
                'status': order.get('friendly_status'),
                'total': order.get('total_amount'),
                'items': len(order.get('items', [])),
                'shipping': order.get('shipping_method'),
                'ordered_on': order.get('created_at')
            }
            if 'delivery_estimate' in order:
                summary['delivery_estimate'] = order['delivery_estimate']
            return summary
        else:
            return result
    
    def order_exists(self, order_id: str) -> bool:
        """Check if order exists"""
        return order_id.strip().upper() in self.orders
    
    def get_all_order_ids(self) -> List[str]:
        """Get all order IDs (for debugging)"""
        return list(self.orders.keys())