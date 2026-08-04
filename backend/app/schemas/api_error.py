from pydantic import BaseModel, ConfigDict


class ApiErrorBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    details: str | None = None
    task_type: str | None = None
    account_id: str | None = None


class ApiErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: ApiErrorBody


STANDARD_ERROR_RESPONSES = {
    400: {
        "model": ApiErrorResponse,
        "description": "Bad Request",
        "content": {
            "application/json": {
                "examples": {
                    "account_not_connected": {
                        "summary": "Account Not Connected",
                        "value": {
                            "error": {
                                "code": "account_not_connected",
                                "message": "Instagram account is not connected",
                                "details": "Instagram account is not connected for this user/account",
                                "task_type": None,
                                "account_id": "test-account-3",
                            }
                        },
                    },
                    "invalid_instagram_reel_link": {
                        "summary": "Invalid Reel Link",
                        "value": {
                            "error": {
                                "code": "invalid_instagram_reel_link",
                                "message": "Invalid Instagram Reel link",
                                "details": "Invalid Instagram Reel link. Provide an instagram.com/reel/... URL.",
                                "task_type": "reel_feedback",
                                "account_id": "test-account-3",
                            }
                        },
                    }
                },
            }
        },
    },
    403: {
        "model": ApiErrorResponse,
        "description": "Forbidden",
        "content": {
            "application/json": {
                "examples": {
                    "reconnect_required": {
                        "summary": "Reconnect Required",
                        "value": {
                            "error": {
                                "code": "reconnect_required",
                                "message": "Instagram account must be reconnected.",
                                "details": "The saved Meta connection is no longer valid.",
                                "task_type": None,
                                "account_id": "test-account-3",
                            }
                        },
                    },
                    "task_not_in_plan": {
                        "summary": "Task Not In Plan",
                        "value": {
                            "error": {
                                "code": "task_not_in_plan",
                                "message": "Task is not available on the current plan",
                                "details": "Task type 'carousel' is not available on the current plan",
                                "task_type": "carousel",
                                "account_id": "test-account-3",
                            }
                        },
                    }
                },
            }
        },
    },
    404: {
        "model": ApiErrorResponse,
        "description": "Not Found",
        "content": {
            "application/json": {
                "examples": {
                    "no_active_account": {
                        "summary": "No Active Account",
                        "value": {
                            "error": {
                                "code": "no_active_account",
                                "message": "No active Instagram account is available",
                                "details": "No connected accounts found for this user",
                                "task_type": None,
                                "account_id": None,
                            }
                        },
                    },
                    "knowledge_pack_not_found": {
                        "summary": "Knowledge Pack Not Found",
                        "value": {
                            "error": {
                                "code": "knowledge_pack_not_found",
                                "message": "Knowledge pack was not found",
                                "details": "Knowledge pack was not found",
                                "task_type": None,
                                "account_id": None,
                            }
                        },
                    },
                    "reel_not_found": {
                        "summary": "Reel Not Found",
                        "value": {
                            "error": {
                                "code": "reel_not_found",
                                "message": "Requested Reel was not found",
                                "details": "Requested Reel was not found in the connected Instagram account",
                                "task_type": "reel_feedback",
                                "account_id": "test-account-3",
                            }
                        },
                    }
                },
            }
        },
    },
    409: {
        "model": ApiErrorResponse,
        "description": "Conflict",
        "content": {
            "application/json": {
                "example": {
                    "error": {
                        "code": "multiple_accounts_require_selection",
                        "message": "Multiple accounts require explicit selection",
                        "details": "Multiple connected accounts found for this user, but no active account is configured. Provide account_id explicitly or set an active account.",
                        "task_type": None,
                        "account_id": None,
                    }
                }
            }
        },
    },
    422: {
        "model": ApiErrorResponse,
        "description": "Validation Failed",
        "content": {
            "application/json": {
                "example": {
                    "error": {
                        "code": "unsupported_task_type",
                        "message": "Unsupported task type",
                        "details": "body.task_type: Input should be 'chat', 'reel_idea', 'reel_script', 'reel_feedback', 'caption', 'carousel', 'profile_audit', 'content_plan', 'link_analysis' or 'performance_summary'",
                        "task_type": "bad_task",
                        "account_id": None,
                    }
                }
            }
        },
    },
    429: {
        "model": ApiErrorResponse,
        "description": "Usage Limit Reached",
        "content": {
            "application/json": {
                "example": {
                    "error": {
                        "code": "generation_limit_reached",
                        "message": "Monthly generation limit reached",
                        "details": "Monthly generation limit reached for current plan (15/15)",
                        "task_type": "caption",
                        "account_id": "test-account-3",
                    }
                }
            }
        },
    },
    502: {
        "model": ApiErrorResponse,
        "description": "Upstream Failure",
        "content": {
            "application/json": {
                "example": {
                    "error": {
                        "code": "meta_api_failed",
                        "message": "Meta API request failed",
                        "details": "Meta Instagram profile request failed",
                        "task_type": None,
                        "account_id": "test-account-3",
                    }
                }
            }
        },
    },
    503: {
        "model": ApiErrorResponse,
        "description": "Temporary AI Rate Limit",
        "content": {
            "application/json": {
                "example": {
                    "error": {
                        "code": "llm_rate_limited",
                        "message": "AI generation is temporarily rate limited. Please try again shortly.",
                        "details": None,
                        "task_type": "reel_idea",
                        "account_id": "test-account-3",
                    }
                }
            }
        },
    },
    500: {
        "model": ApiErrorResponse,
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {
                    "error": {
                        "code": "internal_error",
                        "message": "Internal server error",
                        "details": None,
                        "task_type": None,
                        "account_id": None,
                    }
                }
            }
        },
    },
}
