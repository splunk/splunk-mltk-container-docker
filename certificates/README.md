Certificate files go here

Example command to create files:
`openssl req -x509 -nodes -days 3650 -newkey rsa:2048 -keyout dltk.key -out dltk.pem -subj "/CN=bobobobobbo"`

Use this command or create your own certificates with the same names for them to be used in the DSDL container build process.

All files in this directory are copied to the container. But only certificate files with the correct names and formats are used.