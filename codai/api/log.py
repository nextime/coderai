"""
Request logging middleware for the codai API.
"""

from fastapi import Request


async def log_requests(request: Request, call_next):
    """Log all incoming requests for debugging."""
    # Import global debug flag from app
    from codai.api.app import global_debug
    
    if request.url.path in ["/v1/chat/completions", "/v1/completions"]:
        body = b""
        body_str = ""
        try:
            body = await request.body()
            body_str = body.decode('utf-8')
            
            # In debug mode, dump the full request
            if global_debug:
                print(f"DEBUG: Request body: {body_str[:500]}...")
        except Exception as e:
            print(f"Error reading request body: {e}")
        
        # Call the next middleware/handler
        response = await call_next(request)
        
        # Log response status
        if global_debug:
            print(f"DEBUG: Response status: {response.status_code}")
        
        return response
    else:
        # For non-chat endpoints, just pass through
        response = await call_next(request)
        return response
