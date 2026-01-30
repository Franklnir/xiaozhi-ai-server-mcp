# Alembic (Manual Migrations)

This project uses raw SQL DDL (not SQLAlchemy models). Alembic is set up for **manual** migrations.

## Create a migration

```
alembic revision -m "add column"
```

Edit the generated file in `alembic/versions/` and add SQL using `op.execute(...)`.

Example:
```
from alembic import op

def upgrade():
    op.execute("ALTER TABLE example ADD COLUMN demo INT NULL")

def downgrade():
    op.execute("ALTER TABLE example DROP COLUMN demo")
```

## Run migrations

```
alembic upgrade head
```

## Notes

- Alembic reads `DATABASE_URL` from environment.
- Keep migrations in version control.
