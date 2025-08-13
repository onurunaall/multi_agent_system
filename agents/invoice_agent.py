from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage

from config import llm, checkpointer, store, db

@tool
def get_invoices_by_customer_sorted_by_date(customer_id: str) -> str:
    """
    Look up all invoices for a customer, sorted by invoice date (descending).
    Args:
        customer_id (str): The customer identifier.
    Returns:
        str: The customer's invoices.
    """
    query = """
        SELECT *
        FROM "Invoice"
        WHERE "CustomerId" = %(customer_id)s
        ORDER BY "InvoiceDate" DESC
    """
    return db.run(query, parameters={"customer_id": int(customer_id)})

@tool
def get_invoices_sorted_by_unit_price(customer_id: str) -> str:
    """
    Retrieve all invoices for a customer, sorted by unit price (highest first).
    Args:
        customer_id (str): The customer identifier.
    Returns:
        str: Invoices with unit-price information.
    """
    query = """
        SELECT "Invoice".*, "InvoiceLine"."UnitPrice"
        FROM "Invoice"
        JOIN "InvoiceLine" ON "Invoice"."InvoiceId" = "InvoiceLine"."InvoiceId"
        WHERE "Invoice"."CustomerId" = %(customer_id)s
        ORDER BY "InvoiceLine"."UnitPrice" DESC
    """
    return db.run(query, parameters={"customer_id": int(customer_id)})

@tool
def get_employee_by_invoice_and_customer(invoice_id: str, customer_id: str) -> str:
    """
    Given an invoice ID and customer ID, return the employee linked to it.
    Args:
        invoice_id (str): The invoice ID.
        customer_id (str): The customer identifier.
    Returns:
        str: Employee information or an explanatory message.
    """
    if not invoice_id or not customer_id:
        return "Both invoice ID and customer ID are required"

    query = """
        SELECT "Employee"."FirstName", "Employee"."Title", "Employee"."Email"
        FROM "Employee"
        JOIN "Customer" ON "Customer"."SupportRepId" = "Employee"."EmployeeId"
        JOIN "Invoice" ON "Invoice"."CustomerId" = "Customer"."CustomerId"
        WHERE "Invoice"."InvoiceId" = %(invoice_id)s
          AND "Invoice"."CustomerId" = %(customer_id)s
    """
    
    try:
        result = db.run(query, parameters={"invoice_id": int(invoice_id), "customer_id": int(customer_id)})
        return result if result and result != "[]" else "No employee information found for that invoice."
    except (ValueError, TypeError):
        return "Invalid invoice or customer ID format."
    except Exception:
        return "Unable to retrieve employee information at this time."

invoice_tools = [get_invoices_by_customer_sorted_by_date,
                 get_invoices_sorted_by_unit_price,
                 get_employee_by_invoice_and_customer]

invoice_subagent_prompt = """
You are the invoice-information specialist. Your goal is to answer invoice-related queries for the customer.
The customer's ID is available in the state under the 'customer_id' key. You must use this ID for all tool calls.

TOOLS
- get_invoices_by_customer_sorted_by_date
- get_invoices_sorted_by_unit_price
- get_employee_by_invoice_and_customer

CORE RESPONSIBILITIES
- Retrieve and explain invoice data (dates, totals, items, employees, etc.) for the provided customer_id.
- Maintain a professional and patient tone.

If the data cannot be retrieved, apologize and ask if the customer would like to search for something else.
"""

def create_invoice_agent():
    """Compile the invoice-information sub-agent."""
    return create_react_agent(
        llm,
        tools=invoice_tools,
        checkpointer=checkpointer,
        name='invoice_agent'
    )
