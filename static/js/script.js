// Translate Button Function

function translateText(){

    let input = document.getElementById("inputText").value;

    let output = document.getElementById("outputText");

    let loading = document.getElementById("loading");


    if(input.trim()==""){

        alert("Please enter text");

        return;

    }


    loading.style.display="block";

    output.value="";


    fetch("/translate",{

        method:"POST",

        headers:{
            "Content-Type":"application/json"
        },

        body:JSON.stringify({

            text:input

        })

    })


    .then(response=>response.json())


    .then(data=>{

        loading.style.display="none";

        output.value=data.translation;

    })


    .catch(error=>{

        loading.style.display="none";

        console.log(error);

        output.value="Error occurred";

    });


}




// Clear Input Text

function clearText(){

    document.getElementById("inputText").value = "";

    document.getElementById("outputText").value = "";

}






// Copy Translation

function copyText(){

    let output = document.getElementById("outputText");


    if(output.value.trim() === ""){

        alert("No translation available!");

        return;

    }


    navigator.clipboard.writeText(output.value);


    alert("Translation copied!");

}







// Example Sentence Buttons

let examples = document.querySelectorAll(".examples button");


examples.forEach(function(button){


    button.addEventListener("click",function(){


        document.getElementById("inputText").value = this.innerText;


    });


});
// ======================================
// Show / Hide Password
// ======================================

function togglePassword(id){

    let password = document.getElementById(id);

    if(password.type==="password"){

        password.type="text";

    }

    else{

        password.type="password";

    }

}



// ======================================
// Character Counter
// ======================================

document.addEventListener("DOMContentLoaded",function(){

    let input=document.getElementById("inputText");

    let counter=document.getElementById("charCount");

    if(input && counter){

        input.addEventListener("input",function(){

            counter.innerHTML=this.value.length+" Characters";

        });

    }

});




// ======================================
// Download Translation
// ======================================

function downloadTranslation(){

    let output=document.getElementById("outputText").value;

    if(output==""){

        alert("No translation available!");

        return;

    }

    let blob=new Blob([output],{type:"text/plain"});

    let link=document.createElement("a");

    link.href=URL.createObjectURL(blob);

    link.download="translation.txt";

    link.click();

}




// ======================================
// Loading Animation
// ======================================

function showLoader(){

    let loader=document.getElementById("loading");

    if(loader){

        loader.style.display="block";

    }

}

function hideLoader(){

    let loader=document.getElementById("loading");

    if(loader){

        loader.style.display="none";

    }

}
// ===============================
// Search Translation History
// ===============================

function searchHistory(){

    let input=document.getElementById("historySearch");

    if(!input) return;

    let filter=input.value.toUpperCase();

    let table=document.getElementById("historyTable");

    let tr=table.getElementsByTagName("tr");

    for(let i=1;i<tr.length;i++){

        let td=tr[i].getElementsByTagName("td");

        let found=false;

        for(let j=0;j<td.length;j++){

            if(td[j].innerHTML.toUpperCase().indexOf(filter)>-1){

                found=true;

            }

        }

        tr[i].style.display=found?"":"none";

    }

}
// Character Counter

document.addEventListener("DOMContentLoaded",function(){

let input=document.getElementById("inputText");

let counter=document.getElementById("charCount");

if(input && counter){

counter.innerHTML=input.value.length+" Characters";

input.addEventListener("input",function(){

counter.innerHTML=this.value.length+" Characters";

});

}

});