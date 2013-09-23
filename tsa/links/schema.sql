CREATE TABLE endpoints (
  id serial PRIMARY KEY,
  parent_id integer references endpoints(id),
  url text NOT NULL,
  status_code int,
  redirect text,
  html text,
  content text,
  created timestamp DEFAULT current_timestamp NOT NULL,
  accessed timestamp,
  timeout timestamp,
  error timestamp,
  UNIQUE(url)
);
