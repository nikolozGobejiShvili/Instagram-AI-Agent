# Instagram Agent V1 Architecture

## 1. Frontend (Tichu React)
Responsible for:
- Agent page / dashboard UI
- Chat interface
- Connect Instagram button
- Account switcher
- Plan/limits display
- Generated results display

## 2. Backend API
Responsible for:
- User authentication
- Subscription / plan logic
- Instagram OAuth flow
- Connected account management
- Usage limits
- Calling Langflow
- Saving generations and logs

## 3. Langflow
Responsible for:
- Main agent orchestration
- Tool calling
- Profile audit flow
- Link analysis flow
- Reel script flow
- Carousel flow
- 30-day content plan flow

## 4. Database
Responsible for storing:
- Users
- Plans / subscriptions
- Connected Instagram accounts
- Synced Instagram profile data
- Posts / reels metadata
- Metrics / insights
- Generated outputs
- Usage logs

## 5. External integrations
- Meta / Instagram API
- Langflow API
- Tichu React frontend
- Backend database

## Request flow
1. User opens agent page in Tichu
2. User connects Instagram account
3. Backend saves connected account
4. User sends request in chat
5. Backend sends request to Langflow
6. Langflow calls needed tools
7. Result returns to backend
8. Backend returns final result to frontend