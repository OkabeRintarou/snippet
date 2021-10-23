all: RUN

RUN: src/tr_app.erl src/tr_sup.erl src/tr_server.erl
	erlc -o ebin src/*.erl


.PHONY:clean
clean:
	@rm ebin/*.beam
