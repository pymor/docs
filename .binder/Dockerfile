FROM zivgitlab.wwu.io/pymor/pymor/binder:iosys-to-methods

# binder wants to set the NB_ vars anyways, so we use it to service both setups
ARG NB_USER
ARG NB_UID

USER root
RUN rm -rf /pymor && useradd -d /pymor --shell /bin/bash -u ${NB_UID} -o -c "" -m ${NB_USER}

ENV USER=${NB_USER} \
    HOME=/pymor

COPY iosys-to-methods/* /pymor/
RUN chown -R ${NB_USER} /pymor
USER ${NB_USER}

ENTRYPOINT []
WORKDIR /pymor
