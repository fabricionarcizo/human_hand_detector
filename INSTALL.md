These instructions were tested for macOS High Sierra Version 10.13.4 on October 07, 2018. Use the linked websites and Terminal to execute the instructions.

## Install Apple Command Line Developer Tools
Trigger the installation process of the Command Line Developer Tools (CLT) by typing this in your terminal and letting macOS install the tools required:
```
xcode-select --install
```
_Hint_: Choose "Get Xcode" to install Xcode and the Command Line Developer Tools from the App Store would be a good idea. In this case, open Xcode for the first time and select the SDK version in `Xcode -> Preferences -> Locations -> Command Line Tools`. I have selected Xcode 10.0 (10A255) and I didn't have any problem to install PyQT5 after that.

## Install Homebrew
Homebrew is a package manager for macOS that will help you install correctly development libraries and then symlinks their files into `/usr/local`.

Install it with the following ruby script:
```
ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

## Install Python 3.6 with Homebrew
The code for this assignment was developed and tested with Python 3.6. Install Python and update its packages manager (i.e. pip, setuptools and wheel) using the following commands:
```
brew install python
pip3 install --upgrade pip setuptools wheel
```
Add Homebrew installed executables and Python scripts to your environment variable `path`. Add the following lines to your `~/.bash_profile`:
```
export HOMEBREW_MAKE_JOBS=4
export PATH=/usr/local/opt/python/libexec/bin:/usr/local/bin:/usr/local/sbin:$PATH
```
IMPORTANT: Restart your terminal after editing your `~/.bash_profile`.

Note that you might need to change the Python path above depending on your installed version. Execute the command `brew info python | grep site-packages` to print the corresponding site-packages folder.

## Install Dependency Libraries with Homebrew
First of all, install the dependency libraries for this project using Homebrew. Just copy paste the following commands into your terminal and wait a couple of minutes:
```
brew install pkg-config
brew install qt
brew cask install qt-creator
brew install opencv
brew install sip
brew install pyqt5
brew cask install xquartz
brew tap beeftornado/rmtree
```
To finalize the installation process of dependency Libraries using Homebrew, execute the following commands:
```
cd /usr/local && sudo chown -R $(whoami) Caskroom Cellar Frameworks Homebrew bin etc include lib opt sbin share var
brew update
brew upgrade
brew doctor
```
IMPORTANT: Check if the command `brew doctor` has returned any error. Try to solve them before executing the next steps.

## Install Dependency Packages with pip
Use the package manager `pip` to install some dependency Python libraries. Just copy paste the following commands into your terminal:
```
pip install imutils
pip install scipy
pip install matplotlib
pip install cython
pip install easygui
pip install scikit-learn
pip install scikit-image
pip install pandas
```

**"That's All Folks!"**
