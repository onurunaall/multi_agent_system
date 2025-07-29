import pytest
from unittest.mock import patch, MagicMock
from agents.invoice_agent import get_invoices_by_customer_sorted_by_date, get_invoices_sorted_by_unit_price, get_employee_by_invoice_and_customer, create_invoice_agent


class TestInvoiceTools:
    """Test cases for invoice agent tools."""

    @patch('agents.invoice_agent.db')
    def test_get_invoices_by_customer_sorted_by_date(self, mock_db):
        """Test getting invoices sorted by date."""
        mock_db.run.return_value = "[{'InvoiceId': 1, 'Total': 10.99}]"

        get_invoices_by_customer_sorted_by_date({"customer_id": "123"})

        mock_db.run.assert_called_once()
        query, params = mock_db.run.call_args[0]
        assert "WHERE CustomerId = ?" in query
        assert "ORDER BY InvoiceDate DESC" in query
        assert params == ["123"]

    @patch('agents.invoice_agent.db')
    def test_get_invoices_sorted_by_unit_price(self, mock_db):
        """Test getting invoices sorted by unit price."""
        mock_db.run.return_value = "[{'InvoiceId': 1, 'UnitPrice': 0.99}]"

        result = get_invoices_sorted_by_unit_price({"customer_id": "123"})

        mock_db.run.assert_called_once()
        query, params = mock_db.run.call_args[0]
        assert "CustomerId = ?" in query
        assert params == ["123"]
        assert "ORDER BY InvoiceLine.UnitPrice DESC" in query

    @patch('agents.invoice_agent.db')
    def test_get_employee_by_invoice_and_customer_found(self, mock_db):
        """Test getting employee info when found."""
        mock_db.run.return_value = "[{'FirstName': 'John', 'Title': 'Sales', 'Email': 'john@test.com'}]"

        result = get_employee_by_invoice_and_customer({"invoice_id": "1", "customer_id": "123"})

        assert result == "[{'FirstName': 'John', 'Title': 'Sales', 'Email': 'john@test.com'}]"

    @patch('agents.invoice_agent.db')
    def test_get_employee_by_invoice_and_customer_not_found(self, mock_db):
        """Test getting employee info when not found."""
        mock_db.run.return_value = ""

        result = get_employee_by_invoice_and_customer({"invoice_id": "999", "customer_id": "123"})

        assert "No employee found" in result
        assert "invoice 999" in result
        assert "customer 123" in result