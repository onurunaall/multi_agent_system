import pytest
import os
from unittest.mock import patch, MagicMock

# Mock environment variables before importing anything from the app
@patch.dict("os.environ", {"DB_USER": "test_user",
                           "DB_PASSWORD": "test_pass",
                           "DB_HOST": "localhost",
                           "DB_PORT": "5432",
                           "DB_NAME": "test_db",
                           "OPENAI_API_KEY": "test_key"})
def test_module_imports():
  "Ensure modules can be imported with mocked environment."""
  from agents.invoice_agent import get_invoices_by_customer_sorted_by_date, get_invoices_sorted_by_unit_price, get_employee_by_invoice_and_customer, create_invoice_agent

class TestInvoiceTools:
  @patch.dict("os.environ", {"DB_USER": "test_user",
                             "DB_PASSWORD": "test_pass",
                             "DB_HOST": "localhost",
                             "DB_PORT": "5432",
                             "DB_NAME": "test_db",
                             "OPENAI_API_KEY": "test_key"})
  @patch('agents.invoice_agent.db')
  def test_get_invoices_by_customer_sorted_by_date(self, mock_db):
      """Test getting invoices sorted by date."""
      from agents.invoice_agent import get_invoices_by_customer_sorted_by_date
      
      mock_db.run.return_value = "[{'InvoiceId': 1, 'Total': 10.99}]"
  
      get_invoices_by_customer_sorted_by_date(customer_id="123")
  
      mock_db.run.assert_called_once()
      query = mock_db.run.call_args.args[0]
      params = mock_db.run.call_args.kwargs['parameters']
      assert 'WHERE "CustomerId" = %(customer_id)s' in query
      assert params == {"customer_id": 123}


  @patch.dict("os.environ", {"DB_USER": "test_user",
                             "DB_PASSWORD": "test_pass",
                             "DB_HOST": "localhost",
                             "DB_PORT": "5432",
                             "DB_NAME": "test_db",
                             "OPENAI_API_KEY": "test_key"})
  @patch('agents.invoice_agent.db')
  def test_get_invoices_sorted_by_unit_price(self, mock_db):
      """Test getting invoices sorted by unit price."""
      from agents.invoice_agent import get_invoices_sorted_by_unit_price
      
      mock_db.run.return_value = "[{'InvoiceId': 1, 'UnitPrice': 0.99}]"
  
      result = get_invoices_sorted_by_unit_price(customer_id="123")
  
      mock_db.run.assert_called_once()
      query = mock_db.run.call_args.args[0]
      params = mock_db.run.call_args.kwargs['parameters']
      assert 'WHERE "Invoice"."CustomerId" = %(customer_id)s' in query
      assert params == {"customer_id": 123}
  
  
  @patch.dict("os.environ", {"DB_USER": "test_user",
                             "DB_PASSWORD": "test_pass",
                             "DB_HOST": "localhost",
                             "DB_PORT": "5432",
                             "DB_NAME": "test_db",
                             "OPENAI_API_KEY": "test_key"})
  @patch('agents.invoice_agent.db')
  def test_get_employee_by_invoice_and_customer_found(self, mock_db):
      """Test getting employee info when found."""
      from agents.invoice_agent import get_employee_by_invoice_and_customer
      
      mock_db.run.return_value = "[{'FirstName': 'John', 'Title': 'Sales', 'Email': 'john@test.com'}]"
  
      result = get_employee_by_invoice_and_customer(invoice_id="1", customer_id="123")
  
      assert result == "[{'FirstName': 'John', 'Title': 'Sales', 'Email': 'john@test.com'}]"
  
  
  @patch.dict("os.environ", {"DB_USER": "test_user",
                             "DB_PASSWORD": "test_pass",
                             "DB_HOST": "localhost",
                             "DB_PORT": "5432",
                             "DB_NAME": "test_db",
                             "OPENAI_API_KEY": "test_key"})
  @patch('agents.invoice_agent.db')
  def test_get_employee_by_invoice_and_customer_not_found(self, mock_db):
      """Test getting employee info when not found."""
      from agents.invoice_agent import get_employee_by_invoice_and_customer
      
      mock_db.run.return_value = ""
  
      result = get_employee_by_invoice_and_customer(invoice_id="999", customer_id="123")
  
      assert "No employee found" in result
      assert "invoice 999" in result
      assert "customer 123" in result
