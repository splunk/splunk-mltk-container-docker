# Certificates

From 5.3.0 no certificate or private key is built into the DSDL images, and this
directory is not part of the image build.

At start the container uses, in this order:

1. the certificate and key given by `API_SSL_CERT` and `API_SSL_KEY` (for example
   files mounted from a Kubernetes secret or a Docker volume),
2. `/dltk/.jupyter/dltk.pem` and `/dltk/.jupyter/dltk.key` if a derived image
   provides them,
3. a self-signed pair generated for this installation into `/srv/.dsdl/tls`
   (valid for one year and renewed automatically before it expires).

To create a certificate for option 1 or 2, for example:

`openssl req -x509 -nodes -days 365 -newkey rsa:3072 -keyout dltk.key -out dltk.pem -subj "/CN=<container host>"`

Never commit key files; `certificates/*.key` and `certificates/*.pem` are ignored by git.
