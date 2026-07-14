// ================= PASSWORD TOGGLE =================

function togglePassword() {

    const password = document.getElementById("password");
    const eye = document.getElementById("eye");

    if (password.type === "password") {

        password.type = "text";
        eye.classList.remove("fa-eye");
        eye.classList.add("fa-eye-slash");

    } else {

        password.type = "password";
        eye.classList.remove("fa-eye-slash");
        eye.classList.add("fa-eye");

    }

}

// ================= CHARACTER COUNTER =================

const input = document.getElementById("inputText");

if (input) {

    input.addEventListener("input", function () {

        document.getElementById("charCount").innerHTML =
            input.value.length + " / 1000";

    });

}

// ================= CLEAR =================

function clearInput() {

    if(document.getElementById("inputText")){

        document.getElementById("inputText").value = "";

        document.getElementById("charCount").innerHTML = "0 / 1000";

    }

    if(document.getElementById("outputText")){

        document.getElementById("outputText").value = "";

    }

}

// ================= COPY =================

function copyText(){

    const output=document.getElementById("outputText");

    output.select();

    output.setSelectionRange(0,99999);

    navigator.clipboard.writeText(output.value);

    alert("Translation Copied Successfully");

}

// ================= LOADING =================

const form=document.querySelector("form");

if(form){

form.addEventListener("submit",function(){

const btn=document.querySelector(".translate-btn");

if(btn){

btn.innerHTML="<i class='fa fa-spinner fa-spin'></i> Translating...";

btn.disabled=true;

}

});

}

// ================= DARK MODE =================

const darkBtn=document.getElementById("darkMode");

if(darkBtn){

darkBtn.addEventListener("click",()=>{

document.body.classList.toggle("dark");

});

}

// ================= SCROLL ANIMATION =================

const cards=document.querySelectorAll(".feature-card");

window.addEventListener("scroll",()=>{

cards.forEach(card=>{

const top=card.getBoundingClientRect().top;

if(top<window.innerHeight-100){

card.classList.add("show");

}

});

});
// ================= SWAP LANGUAGE =================

function swapLanguage(){

    const source = document.getElementById("sourceLang");
    const target = document.getElementById("targetLang");

    if(source && target){

        let temp = source.value;

        source.value = target.value;

        target.value = temp;

    }

}


// ================= VOICE INPUT =================

function startListening(){

    if('webkitSpeechRecognition' in window){

        const recognition = new webkitSpeechRecognition();

        recognition.lang="kn-IN";

        recognition.start();


        recognition.onresult=function(event){

            document.getElementById("inputText").value =
            event.results[0][0].transcript;


            document.getElementById("charCount").innerHTML =
            document.getElementById("inputText").value.length + " / 1000";

        };


    }

    else{

        alert("Voice input is not supported");

    }

}