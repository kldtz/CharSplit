#!/bin/bash
set -e

# This script is used for local development container.

# initialize DB if starting with empty dir
if [ ! -f /eb_files/mysql/ibdata1 ]; then
    mysql_install_db
fi

# start mysql
service mysql start

# if needs seeding
if [ ! -d "/eb_files/matters" ]; then
  mysql_install_db --user=mysql

  # restore database
  mysql -uroot -hlocalhost < /src/sql/db.sql
  mysql -uroot -hlocalhost ebrevia < /src/sql/schema.sql
  mysql -uroot -hlocalhost ebrevia < /src/sql/sample_data.sql

  cp -a /seed/eb_files/* /eb_files/
fi

chown tomcat7:tomcat7 /eb_files

# start postfix
service postfix start

# run python app
(cd /src/python && nohup gunicorn --timeout 600 app:app > /tmp/gunicorn.log 2>&1 &)

# add tomcat7 variables
echo "export CATALINA_OPTS=\"-Xmx${TOMCAT_XMX}\"" >> /usr/share/tomcat7/bin/setenv.sh
echo "export ONPREM=\"${ONPREM}\"" >> /usr/share/tomcat7/bin/setenv.sh

# build java app for nginx first load
if [ ! -d "/src/target/ebrevia-extractor" ]; then
  (cd /src && mvn clean compile war:exploded)
fi
if [ ! -d "/var/lib/tomcat7/webapps/api" ]; then
  echo '* Deploying java api...'
  cp -r /src/target/ebrevia-extractor /var/lib/tomcat7/webapps/api
fi

service tomcat7 start

# build static web for nginx first load
if [ ! -d "/src/frontend/dist" ]; then
  echo '* Building frontend dist...'
  cp -r /tmp/node_modules /src/frontend/
  cp -r /tmp/bower_components /src/frontend/
  (cd /src/frontend && grunt build)
  echo '  ...done.'
fi

nginx -g "daemon off;"
