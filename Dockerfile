# run docker
# docker build ./

FROM python:3.7.4-buster
RUN cd ~
RUN git clone https://github.com/xhajnal/mpm.git
RUN cd mpm
RUN pip3 install --user -v .

