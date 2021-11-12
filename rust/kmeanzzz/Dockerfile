FROM rust:1.53

WORKDIR kmeanzz
ADD src src
COPY Cargo.toml .

RUN cargo build --release

CMD ["cargo", "run"]
