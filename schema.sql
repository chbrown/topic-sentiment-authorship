-- $ dropdb tsa; createdb tsa && psql tsa < schema.sql

CREATE TABLE sources (
  id serial PRIMARY KEY,

  name text UNIQUE NOT NULL,
  filepath text,

  created timestamp with time zone DEFAULT current_timestamp NOT NULL
);
CREATE TABLE documents (
  id serial PRIMARY KEY,

  source_id integer references sources(id) NOT NULL,

  label text,
  document text NOT NULL,
  published timestamp,
  details json,

  created timestamp with time zone DEFAULT current_timestamp NOT NULL
);
CREATE INDEX ON documents(source_id, label);


CREATE TABLE endpoints (
  id serial PRIMARY KEY,

  parent_id integer references endpoints(id),
  url text NOT NULL UNIQUE,
  status_code int,
  redirect text,
  html text,
  content text,

  accessed timestamp,
  timeout timestamp
  error timestamp,

  created timestamp DEFAULT current_timestamp NOT NULL,
);
