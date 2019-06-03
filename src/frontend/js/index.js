// disable context field if checkbox not checked
// set required attribute as true if checkbox is checked
var chk = document.getElementById("custom_context_id");
var context = document.getElementById("context_id");
chk.addEventListener("click", enable);
function enable() {
    context.disabled = !this.checked;
    context.required = true;
}

// replace validation bubble with red warning
function replaceValidationUI( form ) {
    form.addEventListener( "invalid", function( event ) {
        event.preventDefault();
    }, true );
    form.addEventListener( "submit", function( event ) {
        if ( !this.checkValidity() ) {
            event.preventDefault();
        }
    });
    var submitButton = form.querySelector( "button:not([type=button]), input[type=submit]" );
    submitButton.addEventListener( "click", function( event ) {
        var invalidFields = form.querySelectorAll( ":invalid" ),
            errorMessages = form.querySelectorAll( ".error-message" ),
            parent;
        for ( var i = 0; i < errorMessages.length; i++ ) {
            errorMessages[ i ].parentNode.removeChild( errorMessages[ i ] );
        }
        for ( var i = 0; i < invalidFields.length; i++ ) {
            parent = invalidFields[ i ].parentNode;
            var message = invalidFields[ i ].validationMessage;
            message = (message === 'Заполните это поле.' ? "Please, fill this field." : "Something went wrong. Please, retype your text.");
            parent.insertAdjacentHTML( "beforeend", "<div class=\"error-message\">" +
                message +
                "</div>" );
        }
        if ( invalidFields.length > 0 ) {
            invalidFields[ 0 ].focus();
        }
    });
}
var forms = document.querySelectorAll( "form" );
for ( var i = 0; i < forms.length; i++ ) {
    replaceValidationUI( forms[ i ] );
}

// send json request
const isValidElement = element => {
  return element.name && element.value && !element.disabled;
};
const isValidValue = element => {
  return (!["checkbox", "radio"].includes(element.type) || element.checked);
};
const formToJSON = elements => [].reduce.call(elements, (data, element) => {
  if (isValidElement(element) && isValidValue(element)) {
      data[element.name] = element.value;
  }
  return data;
}, {});

const handleFormSubmit = event => {
  event.preventDefault();
  const data = formToJSON(form.elements);

  document.getElementById("load").style.visibility="visible";
  var xhr = new XMLHttpRequest();
  if (xhr) {
      xhr.open("POST", "https://seann.ru/api/search", true);
      xhr.setRequestHeader("Content-Type", "application/json");
      xhr.onreadystatechange = function (e) {
        if (xhr.readyState === 4) {
          document.getElementById("load").style.visibility="hidden";
          document.getElementById("question_page_id").style.display="none";
          document.getElementById("answer_page_id").style.display="inline";
          if (xhr.status === 200) {
            console.log(xhr.responseText);
            show_answers(JSON.parse(xhr.responseText));
          } else {
            console.error(xhr.statusText);
          }
        }
      };
      xhr.onerror = function (e) {
        console.error(xhr.statusText);
      };
      xhr.send(JSON.stringify(data));
  };
};

function show_answers(response) {
  var paragraph = document.getElementById("question_div");
  paragraph.style.display = "block";
  var paragraph = document.getElementById("question");
  paragraph.textContent = response["query"];

  if (response["answers"].length === 0) {
    var page = document.getElementById("answer_page_id");
    page.innerHTML += "<div class=\"answer_block\"><div class=\"answer_score\">" +
      "<p>Oops..</p></div>" +
      "<div class=\"context_answer_page\" id=\"context0\"></div></div>";
    var paragraph = document.getElementById("context0");
    paragraph.style.display = "block";
    paragraph.innerHTML = "<p id=\"answer0\"></p>";
    var paragraph = document.getElementById("answer0");
    paragraph.innerHTML = "Nothing is found on your request.";
  } else {
    var page = document.getElementById("answer_page_id");
    page.innerHTML += "<div class=\"legend\">\n" +
      "<div class=\"legend_answer\"></div> Answer   \n" +
      "<div class=\"legend_panswer\"></div> Plausible Answer   \n" +
      "<div class=\"legend_cross\"></div> Intersection of Answer and Plausible Answer\n" +
      "</div></br>\n";
    for (var i = 0; i < response["answers"].length; i++) {
      var page = document.getElementById("answer_page_id");
      page.innerHTML += "<div class=\"answer_block\"><div class=\"answer_score\"><p>Confidence score: " +
        (100 * response["answers"][i]["has_ans_score"]).toString().slice(0, 4) +
        "%</p></div><div class=\"context_answer_page\" id=\"context" + (i).toString() + "\"></div></div>";

      var paragraph = document.getElementById("context" + (i).toString());
      paragraph.style.display = "block";
      paragraph.innerHTML = "<p id=\"answer" + (i).toString() + "\"></p>";

      var paragraph = document.getElementById("answer" + (i).toString());
      paragraph.innerHTML = tag_answers(response["answers"][i]["text"],
       response["answers"][i]["start_offset"],
       response["answers"][i]["end_offset"],
       response["answers"][i]["start_poffset"],
       response["answers"][i]["end_poffset"]);
    }
  }
}

function tag_answers(context, ans_begin, ans_end, pans_begin, pans_end) {
  if ((ans_begin < pans_end) && (pans_begin >= ans_end)) {
    return context.slice(0, ans_begin) + "<mark class=\"mark_answer\">" +
      context.slice(ans_begin, ans_end) + "</mark>" +
      context.slice(ans_end, pans_begin) + "<mark class=\"mark_panswer\">" +
      context.slice(pans_begin, pans_end) + "</mark>" +
      context.slice(pans_end);
  } else if ((ans_begin >= pans_end) && (pans_begin < ans_end)) {
    return context.slice(0, pans_begin) + "<mark class=\"mark_panswer\">" +
       context.slice(pans_begin, pans_end) + "</mark>" +
       context.slice(pans_end, ans_begin) + "<mark class=\"mark_answer\">" +
       context.slice(ans_begin, ans_end) + "</mark>" +
       context.slice(ans_end);
  } else if ((ans_begin === pans_begin) && (ans_end < pans_end)) {
    return context.slice(0, pans_begin) + "<mark class=\"mark_cross\">" +
       context.slice(pans_begin, ans_end) + "</mark><mark class=\"mark_panswer\">" +
       context.slice(ans_end, pans_end) + "</mark>" +
       context.slice(pans_end);
  } else if ((ans_begin === pans_begin) && (pans_end < ans_end)) {
    return context.slice(0, pans_begin) + "<mark class=\"mark_cross\">" +
       context.slice(pans_begin, ans_end) + "</mark><mark class=\"mark_answer\">" +
       context.slice(ans_end, pans_end) + "</mark>" +
       context.slice(pans_end);
  } else if ((ans_begin === pans_begin) && (ans_end === pans_end)) {
    return context.slice(0, pans_begin) + "<mark class=\"mark_cross\">" +
       context.slice(pans_begin, ans_end) + "</mark>" +
       context.slice(ans_end);
  } else if ((ans_end === pans_end) && (ans_begin < pans_begin)) {
    return context.slice(0, ans_begin) + "<mark class=\"mark_answer\">" +
       context.slice(ans_begin, pans_begin) + "</mark><mark class=\"mark_cross\">" +
       context.slice(pans_begin, pans_end) + "</mark>" +
       context.slice(pans_end);
  } else if ((ans_end === pans_end) && (ans_begin > pans_begin)) {
    return context.slice(0, pans_begin) + "<mark class=\"mark_panswer\">" +
       context.slice(pans_begin, ans_begin) + "</mark><mark class=\"mark_cross\">" +
       context.slice(ans_begin, pans_end) + "</mark>" +
       context.slice(pans_end);
  } else if ((ans_begin < pans_end) && (pans_begin < ans_end) && (ans_begin < pans_begin)) {
    return context.slice(0, ans_begin) + "<mark class=\"mark_answer\">" +
       context.slice(ans_begin, pans_begin) + "</mark><mark class=\"mark_cross\">" +
       context.slice(pans_begin, ans_end) + "</mark><mark class=\"mark_panswer\">" +
       context.slice(ans_end, pans_end) + "</mark>" +
       context.slice(pans_end);
  } else if ((ans_begin < pans_end) && (pans_begin < ans_end) && (ans_begin > pans_begin)) {
    return context.slice(0, pans_begin) + "<mark class=\"mark_panswer\">" +
       context.slice(pans_begin, ans_begin) + "</mark><mark class=\"mark_cross\">" +
       context.slice(ans_begin, pans_end) + "</mark><mark class=\"mark_answer\">" +
       context.slice(pans_end, ans_end) + "</mark>" +
       context.slice(ans_end);
  } else if ((ans_begin < pans_begin) && (pans_end < ans_end)) {
    return context.slice(0, ans_begin) + "<mark class=\"mark_answer\">" +
       context.slice(ans_begin, pans_begin) + "</mark><mark class=\"mark_cross\">" +
       context.slice(pans_begin, pans_end) + "</mark><mark class=\"mark_answer\">" +
       context.slice(pans_end, ans_end) + "</mark>" +
       context.slice(ans_end);
  } else if ((pans_begin < ans_begin) && (ans_end < pans_end)) {
    return context.slice(0, pans_begin) + "<mark class=\"mark_panswer\">" +
       context.slice(pans_begin, ans_begin) + "</mark><mark class=\"mark_cross\">" +
       context.slice(ans_begin, ans_end) + "</mark><mark class=\"mark_panswer\">" +
       context.slice(ans_end, pans_end) + "</mark>" +
       context.slice(pans_end);
  }
}

const form = document.getElementById( "custom_request_id" );
form.addEventListener("submit", handleFormSubmit);
