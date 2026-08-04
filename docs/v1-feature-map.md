# V1 Feature Map

## 1. Connect Instagram
### Frontend
- "Connect Instagram" button
- Connected account status

### Backend
- Start OAuth flow
- Handle OAuth callback
- Save connected account
- Save tokens securely

### Langflow
- Not used directly

### Required data
- User ID
- Instagram account ID
- Access token / refresh data

### Output
- Connected account saved successfully


## 2. Account Switcher
### Frontend
- Dropdown / switcher for connected accounts

### Backend
- Return connected accounts for current user
- Save selected active account

### Langflow
- Receives selected account ID in requests

### Required data
- User ID
- Connected accounts list
- Active account ID

### Output
- Active Instagram account selected


## 3. Profile Audit
### Frontend
- "Audit profile" action/button
- Audit result screen

### Backend
- Validate account access
- Send request to Langflow
- Save generated result

### Langflow
- Profile audit flow
- Uses profile/account data

### Required data
- Profile info
- Bio
- Category
- Posts/reels overview

### Output
- Structured profile audit


## 4. Reel/Post Link Analysis
### Frontend
- Input field for Instagram link
- Analyze button

### Backend
- Validate input
- Send request to Langflow
- Save result

### Langflow
- Link analysis flow

### Required data
- Instagram URL
- Optional selected account context

### Output
- Analysis + adaptation suggestions


## 5. Reel Script Generation
### Frontend
- Prompt input
- Generate script action

### Backend
- Send request to Langflow
- Save result
- Check usage limits

### Langflow
- Reel script flow

### Required data
- User prompt
- Selected account context
- Optional recent performance data

### Output
- Reel script with hook, structure, CTA


## 6. Carousel Text Generation
### Frontend
- Prompt input
- Generate carousel action

### Backend
- Send request to Langflow
- Save result
- Check usage limits

### Langflow
- Carousel flow

### Required data
- User prompt
- Selected account context

### Output
- Slide-by-slide carousel text


## 7. 30-Day Content Plan
### Frontend
- Generate plan action
- Plan results view

### Backend
- Send request to Langflow
- Save result
- Check usage limits

### Langflow
- 30-day plan flow

### Required data
- Selected account context
- Optional niche/category data
- Recent content summary

### Output
- 30-day structured content plan


## 8. Recent Content Performance Summary
### Frontend
- "Recent performance" action/card

### Backend
- Read synced Instagram data
- Prepare summary request for Langflow or direct backend summary

### Langflow
- Optional summary flow

### Required data
- Recent posts/reels
- Metrics / insights

### Output
- Summary of what worked / what did not