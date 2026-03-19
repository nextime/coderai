"""
Request logging middleware for the codai API.
"""

import json
from fastapi import Request


async def log_requests(request: Request, call_next):
    """Log all incoming requests for debugging."""
    # Import global debug flag from state
    from codai.api.state import get_global_debug
    global_debug = get_global_debug()
    
    if request.url.path in ["/v1/chat/completions", "/v1/completions"]:
        body = b""
        body_str = ""
        try:
            body = await request.body()
            body_str = body.decode('utf-8')
            
            # In debug mode, dump the full request
            if global_debug:
                print(f"\n{'='*80}")
                print(f"=== FULL REQUEST DEBUG ===")
                print(f"{'='*80}")
                print(f"Method: {request.method}")
                print(f"URL: {request.url}")
                print(f"Headers:")
                for k, v in request.headers.items():
                    print(f"  {k}: {v}")
                print(f"\n--- Body ---")
                # Print full body without truncation
                try:
                    # Try to pretty-print JSON
                    parsed = json.loads(body_str)
                    print(json.dumps(parsed, indent=2))
                except:
                    # If not JSON, print as-is
                    print(body_str)
                print(f"{'='*80}\n")
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
