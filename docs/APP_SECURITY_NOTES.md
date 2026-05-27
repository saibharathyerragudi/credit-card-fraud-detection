# App Security Notes

The original Flask app included local user signup logic, a SQLite database, OTP email flow, and a hardcoded email credential. Those pieces were removed from the public portfolio version.

## What Changed

- Removed hardcoded Gmail credentials.
- Removed the tracked `signup.db` database.
- Removed authentication pages that depended on missing templates and local user data.
- Replaced the app with a small JSON inference API.
- Added `.gitignore` rules for local databases, raw data, model artifacts, and environment files.

## Current API Surface

- `GET /`: project metadata
- `GET /health`: service health and model availability
- `GET /metadata`: expected feature schema
- `POST /predict`: fraud prediction from JSON features

## Production Notes

A production fraud application should never commit secrets, customer data, local databases, or trained models with sensitive metadata. Use environment variables, secret managers, access controls, audit logging, and secure model artifact storage.
