"""Shopify Admin token for the daily jobs: client-credentials grant with the jewel-vs app (secrets SHOPIFY_CLIENT_ID / SHOPIFY_CLIENT_SECRET).
  python eval/shop_token.py   -> checks the token works and which scopes it has (prints no secret)."""
import os, requests
SHOP = os.environ.get("SHOP", "jewelaccessories.myshopify.com")
def token():
    r = requests.post(f"https://{SHOP}/admin/oauth/access_token", timeout=30, data={
        "client_id": os.environ["SHOPIFY_CLIENT_ID"], "client_secret": os.environ["SHOPIFY_CLIENT_SECRET"], "grant_type": "client_credentials"})
    r.raise_for_status(); return r.json()["access_token"]
def gql(tok, q, v=None):
    r = requests.post(f"https://{SHOP}/admin/api/2026-10/graphql.json", json={"query": q, "variables": v or {}}, headers={"X-Shopify-Access-Token": tok}, timeout=60)
    r.raise_for_status(); return r.json()
if __name__ == "__main__":
    for k in ("SHOPIFY_CLIENT_ID", "SHOPIFY_CLIENT_SECRET"): print(k, "set" if os.environ.get(k) else "MISSING")
    t = token(); print("token ok, length", len(t))
    print(gql(t, "{ shop { name } currentAppInstallation { accessScopes { handle } } products(first:1){ nodes { title } } }"))
