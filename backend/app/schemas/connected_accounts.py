from pydantic import BaseModel


class ConnectedAccountItem(BaseModel):
    account_id: str
    label: str
    niche: str
    is_active: bool


class ConnectedAccountsSavePayload(BaseModel):
    user_id: str
    accounts: list[ConnectedAccountItem]


class SetActiveAccountPayload(BaseModel):
    account_id: str


class ConnectedAccountsResponse(BaseModel):
    user_id: str
    accounts: list[ConnectedAccountItem]
