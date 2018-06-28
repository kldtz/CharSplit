# eBrevia-extractor - Instructions for Docker


## Build base image (`Dockerfile`)

This is the base image used for both local development and production. There are 2 options:

1. Setup an automated build on [Docker Hub](https://hub.docker.com/) linking to [eBrevia/extractor](https://github.com/eBrevia/extractor) repository

2. Locally build the image from project folder, run `docker build -t <image_name> .` 


## Setup local development container (`Dockerfile_dev`)

1. Install docker and docker-compose on your machine

2. Run `docker login` if using base image from docker hub

3. From `docker/compose` folder, run `docker-compose -f dev.yml up -d`

4. **Switch source code to working branch**

5. To rebuild backend, run `docker-compose -f dev.yml exec extractor bash -c "cd /src && mvn clean compile war:exploded"`

6. To deploy backend, run `cp -r /src/target/ebrevia-extractor /var/lib/tomcat7/webapps/api`

7. To rebuild frontend, run `docker-compose -f dev.yml exec extractor bash -c "cd /src/frontend && grunt build"`.
Browse application at: [http://localhost/login.html](http://localhost/login.html).

8. To start grunt server, run `docker-compose -f dev.yml exec extractor bash -c "cd /src/frontend && grunt serve"`.
Browse application at: [http://localhost:9000/login.html](http://localhost:9000/login.html).

9. To debug backend using tomcat7-maven-plugin:
  - Stop tomcat7 service, run `docker-compose -f dev.yml exec extractor service tomcat7 stop`
  - Run `docker-compose -f dev.yml exec extractor bash -c "cd /src && mvn clean compile -Pdevelopment"`

10. To enter into the running docker container, run `docker-compose -f dev.yml exec extractor bash`. 
Inside the docker container, we can perform those above commands without having `docker-compose -f dev.yml exec extractor bash -c`.
  - Rebuild backend: `cd /src && mvn clean compile war:exploded`
  - Rebuild frontend: `cd /src/frontend && grunt build`

11. To start local development service whenever the machine reboots, run `docker-compose -f dev.yml start`


## Build production image (`Dockerfile_prod`)

1. Switch source code to deploying branch

2. Start **local development** container and rebuild backend & frontend

3. From project directory, run `docker build -f Dockerfile_prod -t <image_name:tag> .`

4. To push image to docker hub, `docker push <image_name>` (`docker login` is required)
