## Link crawling requirements

Create the database (we'll call it `links`):

    dropdb links
    createdb links
    psql links < schema.sql

The table looks like this:

| column | type | special |
|:-------|:-----|:--------|
| id | integer | PK |
| parent_id | integer | FK -> endpoints(id) |
| url | text | UNIQUE, NOT NULL |
| status_code | int | |
| redirect | text | |
| html | text | |
| content | text | |
| created | timestamp | DEFAULT current_timestamp NOT NULL |
| accessed | timestamp | |
| timeout | timestamp | |
| error | timestamp | |
