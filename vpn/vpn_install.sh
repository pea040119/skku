#!/bin/sh

# $1: [PACKAGE PATH]
# $2: [INTERACTIVE OPTION]

INSTALL_PATH="/usr/local/axgate"
PACKAGE_PATH="./"
PACKAGE_FILE="slvpn_pkg_x86_64.tgz"
CONF_FILE="/etc/sslvpn_auto.conf"
LIB_FILE="/usr/lib/libMagicCrypto.so"
SLEEP_TIME=5
INTERACTIVE=1

if [ $# -ge 1 ]
then
  if [ -d $1 ]
  then
    PACKAGE_PATH=$1
  else
    echo "해당경로가 존재하지 않습니다. ""$1"
    exit
  fi
fi

if [ $# -ge 2 ]
then
  if [ $2 = "-nointeractive" ]
  then
    INTERACTIVE=0
  elif [ $2 = "-interactive" ]
  then
    INTERACTIVE=1
  else
    INTERACTIVE=1
  fi
fi

if [ "$INTERACTIVE" -eq 0 ]
then
  echo "설치시간: "$(date)
fi

if [ $(id -u) -ne 0 ]
then
  echo "vpn 설치를 위하여는 root 권한이 필요합니다."
  exit
fi

MACHINE_TYPE=`uname -m`
if [ -n "$MACHINE_TYPE" ]
then
  #MACHINE_ARM="${MACHINE_TYPE:0:3}" #no support dash
  MACHINE_ARM=$(echo "$MACHINE_TYPE" | cut -c 1-3)
  if [ "$MACHINE_ARM" = "arm" ]
  then
    PACKAGE_FILE="$PACKAGE_PATH""/sslvpn_pkg_""$MACHINE_ARM"".tgz"
  else
    PACKAGE_FILE="$PACKAGE_PATH""/sslvpn_pkg_""$MACHINE_TYPE"".tgz"
  fi
else
  echo "os 의 machine 을 확인할 수 없습니다."
  exit
fi

if [ ! -f "$PACKAGE_FILE" ] 
then
  echo "파일이 현재경로에 존재하지 않습니다. ""$PACKAGE_FILE"
  exit
fi

VPN_PID_CHECK=`"$INSTALL_PATH"/busybox pidof sslvpnd > /dev/null 2>&1; echo $?`
if [ "$VPN_PID_CHECK" -eq 0 ]
then
  VPN_PID=`"$INSTALL_PATH"/busybox pidof sslvpnd`
else
  VPN_PID_CHECK=`pidof sslvpnd > /dev/null 2>&1; echo $?`
  if [ "$VPN_PID_CHECK" -eq 0 ]
  then
    VPN_PID=`pidof sslvpnd`
  fi
fi

if [ "$INTERACTIVE" -eq 0 ]
then
  if [ -n "$VPN_PID" ]
  then
    kill -2 "$VPN_PID"
    sleep "$SLEEP_TIME"
    kill -9 "$VPN_PID" > /dev/null 2>&1
  fi
else
  while [ -n "$VPN_PID" ]
  do
    read -p "vpn 이 실행중입니다. 종료하시겠습니까? (y/n)" yn
    case $yn in
      [Yy]* ) echo "vpn 종료 중입니다.";
        kill -2 "$VPN_PID"
        sleep "$SLEEP_TIME"
        kill -9 "$VPN_PID" > /dev/null 2>&1
        break;;
      [Nn]* ) echo "";
        exit;;
      *) echo "y 나n 을 선택하여 주세요.";;
    esac
  done
fi

if [ -d "$INSTALL_PATH" ]
then
  rm -f "$INSTALL_PATH"/*
else
  mkdir -p "$INSTALL_PATH"
fi

if [ -f "$PACKAGE_FILE" ] 
then
  tar xzf "$PACKAGE_FILE" -C $INSTALL_PATH
else
  echo "파일이 현재경로에 존재하지 않습니다. ""$PACKAGE_FILE"
  exit
fi

if [ -f "$CONF_FILE" ]
then
  rm -f "$INSTALL_PATH"/sslvpn_auto.conf
else
  mv -f "$INSTALL_PATH"/sslvpn_auto.conf "$CONF_FILE"
fi

if [ -f "$LIB_FILE" ]
then
  rm -f "$INSTALL_PATH"/libMagicCrypto.so > /dev/null 2>&1
else
  mv -f "$INSTALL_PATH"/libMagicCrypto.so "$LIB_FILE" > /dev/null 2>&1
  ldconfig -v > /dev/null 2>&1
fi

"$INSTALL_PATH"/sslvpnd

echo "vpn 설치가 완료 되었습니다."

