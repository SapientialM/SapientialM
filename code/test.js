var __readline = require('readline-sync');
var readline = __readline.prompt;
var printsth = console.log;

let line = "";
while(line = readline()){
    printsth(line);
}