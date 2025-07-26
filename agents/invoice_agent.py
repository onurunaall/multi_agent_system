from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from config import llm, checkpointer, store, db
from schemas import State

@tool
def get_invoices_by_customer_sorted_by_date(customer_id: str) -> list[dict]:
    """
    Look up all invoices for a customer, sorted by invoice date (descending).
    Args:
        customer_id (str): The customer identifier.
    Returns:
        list[dict]: The customer's invoices.
    """
    return db.run(
        """
        SELECT *
        FROM Invoice
        WHERE CustomerId = ?
        ORDER BY InvoiceDate DESC;
        """,
        [customer_id]
    )


@tool
def get_invoices_sorted_by_unit_price(customer_id: str) -> list[dict]:
    """
    Retrieve all invoices for a customer, sorted by unit price (highest first).
    Args:
        customer_id (str): The customer identifier.
    Returns:
        list[dict]: Invoices with unit-price information.
    """
    query = """
        SELECT Invoice.*, InvoiceLine.UnitPrice
        FROM Invoice
        JOIN InvoiceLine ON Invoice.InvoiceId = InvoiceLine.InvoiceId
        WHERE Invoice.CustomerId = ?
        ORDER BY InvoiceLine.UnitPrice DESC;
    """
    return db.run(query, [customer_id])


@tool
def get_employee_by_invoice_and_customer(invoice_id: str, customer_id: str) -> dict:
    """
    Given an invoice ID and customer ID, return the employee linked to it.
    Args:
        invoice_id (int): The invoice ID.
        customer_id (str): The customer identifier.
    Returns:
        dict: Employee information or an explanatory message.
    """
    if not invoice_id or not customer_id:
        return "Both invoice ID and customer ID are required"
        
    query = """
        SELECT Employee.FirstName, Employee.Title, Employee.Email
        FROM Employee
        JOIN Customer ON Customer.SupportRepId = Employee.EmployeeId
        JOIN Invoice ON Invoice.CustomerId = Customer.CustomerId
        WHERE Invoice.InvoiceId = ?
          AND Invoice.CustomerId = ?;
    """
    result = db.run(query, [invoice_id, customer_id], include_columns=True)
    return (
        result
        if result
        else f"No employee found for invoice {invoice_id} and customer {customer_id}."
    )

invoice_tools = [get_invoices_by_customer_sorted_by_date,
                 get_invoices_sorted_by_unit_price,
                 get_employee_by_invoice_and_customer]

invoice_subagent_prompt = """
You are the invoice-information specialist. Only respond to invoice-related queries.

TOOLS
- get_invoices_by_customer_sorted_by_date
- get_invoices_sorted_by_unit_price
- get_employee_by_invoice_and_customer

CORE RESPONSIBILITIES
- Retrieve and explain invoice data (dates, totals, items, employees, etc.)
- Maintain a professional and patient tone

If the data cannot be retrieved, apologise and ask if the customer would like to search something else.
"""

def create_invoice_agent():
    """Compile the invoice-information sub-agent."""
    return create_react_agent(llm,
                              tools=invoice_tools,
                              name="invoice_information_subagent",
                              prompt=invoice_subagent_prompt,
                              state_schema=State,
                              checkpointer=checkpointer,
                              store=store)
