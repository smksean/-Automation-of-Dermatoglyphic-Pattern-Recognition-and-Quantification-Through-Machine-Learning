# Private Streamlit Subtype Annotation App

This application gives an authorized expert a browser-based workflow for the
672 non-holdout arch and whorl images that still need subtype labels. Images are
stored in a private Supabase Storage bucket, annotations are written to
Postgres, and a current CSV is regenerated in private storage after every save.

Before any cloud deployment, the same interface can run entirely on localhost
against the existing private package. In local mode, images and annotations do
not leave the computer.

## Localhost preview first

Install the tested dependencies into the project-local environment:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Copy the local example configuration:

```powershell
Copy-Item .streamlit\secrets.local.example.toml .streamlit\secrets.toml
```

Then run:

```powershell
.venv\Scripts\streamlit.exe run annotation_app\streamlit_app.py
```

Open `http://localhost:8501`, then use the example credentials from the local
secrets file:

- reviewer ID: `local-reviewer`
- access code: `local-preview`

Local test annotations are written to the ignored folder
`annotation_exports/local_preview/`. Delete or move that folder when the local
test is finished and before beginning real expert annotation.

## Security architecture

- GitHub contains application code only—never biometric images or credentials.
- Streamlit Community Cloud must be configured as a private app with only named
  viewers.
- A second project access code is required unless Streamlit OIDC is configured.
- Supabase Storage must use a private bucket.
- The Streamlit server uses the Supabase service-role secret from Streamlit's
  secrets manager. The key is never sent to the browser or committed to Git.
- Database tables use row-level security with no browser-role policies.
- Every annotation revision is retained in `annotation_events`.
- Concurrent edits are rejected through revision checking instead of silently
  overwriting another reviewer's work.

Cloud storage of biometric data must be authorized by the institution's ethics,
privacy, and data-governance process before deployment.

## Cloud deployment after local approval

Only continue with the following sections after the localhost workflow has been
reviewed and cloud processing of the biometric data has been approved.

## 1. Create and link the Supabase backend

1. Create a project in the institution-approved account and region.
2. Authenticate and link this repository with the pinned CLI:

```powershell
npx supabase login
npx supabase link --project-ref YOUR_PROJECT_REFERENCE
npx supabase db push --dry-run
npx supabase db push
```

3. The migration in `supabase/migrations/` creates the database objects and the
   private `fingerprint-review` bucket. Do not add public read policies.
4. As a manual alternative, run [`supabase_schema.sql`](supabase_schema.sql) in
   the dashboard SQL editor.
5. Copy the project URL and a server-side Supabase secret key. Never use the
   secret key in browser code or commit it to Git.

Supabase private buckets require authenticated access or time-limited signed
URLs. This app downloads images on the trusted Streamlit server and sends only
the selected image bytes to an authorized session.

## 2. Upload the review package

Install dependencies in a private administrative environment:

```powershell
python -m pip install -r requirements.txt
```

Set credentials for the current PowerShell session:

```powershell
$env:SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
$env:SUPABASE_SERVICE_ROLE_KEY = "YOUR_SERVER_SIDE_SERVICE_ROLE_KEY"
$env:SUPABASE_BUCKET = "fingerprint-review"
```

Validate without uploading:

```powershell
python scripts/upload_subtype_review_to_supabase.py --dry-run
```

Upload the 672 validated images and static review records:

```powershell
python scripts/upload_subtype_review_to_supabase.py
```

The uploader checks the local package's SHA-256 manifest before transferring
anything. Re-running it safely refreshes the same image paths and static review
rows; it does not delete annotations.

## 3. Configure Streamlit secrets

For a local administrative test, copy
[`secrets.example.toml`](../.streamlit/secrets.example.toml) to
`.streamlit/secrets.toml` and replace the placeholders. That destination is
ignored by Git.

Use a long, randomly generated project access code. Keep
`allowed_reviewer_ids` limited to the professor and named adjudicators.

For stronger identity-based authentication, configure Streamlit's built-in
OIDC support with an approved Google or Microsoft identity provider, set
`use_oidc=true`, and put the permitted email addresses in
`allowed_reviewer_ids`.

## 4. Test locally as the administrator

```powershell
streamlit run annotation_app/streamlit_app.py
```

Confirm that:

- the access gate rejects an incorrect code;
- private images load without exposing a public Storage URL;
- arch images only offer arch subtypes;
- whorl images only offer whorl subtypes;
- unclear labels cannot be accepted;
- adjudications and exclusions require notes;
- saving updates the progress counter and downloadable CSV;
- `annotation_events` receives an audit row.

## 5. Deploy privately on Streamlit Community Cloud

1. Push the code—but not `.streamlit/secrets.toml` or any review images—to
   GitHub.
2. In Streamlit Community Cloud, create an app from this repository.
3. Set the entrypoint to `annotation_app/streamlit_app.py`.
4. Paste the real secret values into the app's **Advanced settings → Secrets**.
5. In **Sharing**, select **Only specific people can view this app**.
6. Invite only the professor and named project administrators by email.
7. Test the deployed link before sending it to the reviewer.

Streamlit private viewers authenticate with Google or single-use emailed links.
The app should never be made public or searchable.

## Reviewer workflow

The professor sees one fingerprint at a time and selects:

- confirmed subtype;
- confidence; and
- an optional main-type concern or note only when the image needs re-checking.

The normal review action is derived automatically. A clear subtype is accepted;
an `unclear` subtype or main-type concern is sent for adjudication.

Each save updates the shared database, appends an immutable audit event, and
refreshes `exports/subtype_labeling_latest.csv` in the private bucket. The
sidebar also provides an immediate CSV download and progress filters.

## Operational safeguards

- Remove access immediately when a reviewer leaves the project.
- Rotate the project access code and Supabase service-role key after exposure.
- Enable account MFA for project administrators.
- Review Supabase and Streamlit access logs periodically.
- Maintain an approved retention and deletion schedule.
- Do not use the 136 locked-holdout images in this app before final evaluation.

## Official documentation

- [Streamlit private app sharing](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app)
- [Streamlit secrets management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)
- [Streamlit authentication](https://docs.streamlit.io/develop/concepts/connections/authentication)
- [Supabase private buckets](https://supabase.com/docs/guides/storage/buckets/fundamentals)
- [Supabase Storage access control](https://supabase.com/docs/guides/storage/security/access-control)
- [Supabase row-level security](https://supabase.com/docs/guides/database/postgres/row-level-security)
