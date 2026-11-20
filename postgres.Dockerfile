FROM postgres:17

RUN apt-get update && apt-get install -y openssl && rm -rf /var/lib/apt/lists/*

RUN openssl req -new -x509 -days 3650 -nodes \
    -out /var/lib/postgresql/server.crt \
    -keyout /var/lib/postgresql/server.key \
    -subj "/CN=localhost" && \
    chown postgres:postgres \
        /var/lib/postgresql/server.crt \
        /var/lib/postgresql/server.key && \
    chmod 600 /var/lib/postgresql/server.key
