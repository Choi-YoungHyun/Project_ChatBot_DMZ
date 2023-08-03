function onTextareaInput() {
    const textarea = document.getElementById('input_sentence');
    const sendButton = document.getElementById('send_btn_img');
    console.log('들어옴');
    // textarea의 입력 내용이 비어있으면 기본 이미지로 변경
    if (textarea.value.trim() === '') {
        sendButton.src = './send.png'; // 기본 이미지 파일명
    } else {
        sendButton.src = './send_on.png'; // 변경된 이미지 파일명
    }
}

function getRandomValue() {
    const header_random1 = document.getElementById('but1');
    const header_random2 = document.getElementById('but2');
    const header_random3 = document.getElementById('but3');
    const test = document.getElementsByClassName('.random')
    console.log(test)



    header_random1.addEventListener(('click'), () => {
        random_click('1')
    });
    header_random2.addEventListener(('click'), () => {
        random_click('2')
    });
    header_random3.addEventListener(('click'), () => {
        random_click('3')
    });

    let xhr = new XMLHttpRequest();
    xhr.open('GET', 'http://127.0.0.1:8000', true);
    xhr.setRequestHeader('Content-Type', 'application/json');

    xhr.onreadystatechange = function () {
        if (xhr.readyState === XMLHttpRequest.DONE) {
            if (xhr.status === 200) {
                const result = JSON.parse(xhr.responseText);
                document.getElementById('but1').innerText = result.random1[0];
                document.getElementById('but2').innerText = result.random2[0];
                document.getElementById('but3').innerText = result.random3[0];
            } else {
                console.error('Error:', xhr.status);
                document.getElementById('but1').innerText = 'Error occurred. Please try again.';
            }
        }
    };
    xhr.send();
}

// textarea 입력 내용이 변경될 때마다 onTextareaInput 함수 호출
document.getElementById('input_sentence').addEventListener('input', onTextareaInput);
document.getElementById('input_sentence').addEventListener('keydown', function(e) {
    if (e.keyCode === 13) {
        send_Sinjoword()
    }
    else if(e.shiftKey && e.keyCode === 13){
        alert('줄바꿈하자')
    }
});


function new_chat(question, answer) {
    const chatZoneDiv = document.getElementById("chat_zone");

    // 첫 번째 자식 요소를 생성합니다.
    const askDiv = document.createElement("div");
    askDiv.className = "ask";

    var askParagraph = document.createElement("p");
    askParagraph.textContent = question
    askDiv.appendChild(askParagraph);

    // chat_zone div에 첫 번째 자식 요소를 추가합니다.
    chatZoneDiv.appendChild(askDiv);

    // 두 번째 자식 요소를 생성합니다.
    const answerContainerDiv = document.createElement("div");
    answerContainerDiv.className = "answer_container";

    const dmzIconDiv = document.createElement("div");
    dmzIconDiv.className = "dmz_icon";

    const dmzIconImg = document.createElement("img");
    dmzIconImg.src = "logo_D.png";
    dmzIconImg.alt = "";
    dmzIconDiv.appendChild(dmzIconImg);

    const dmzIconParagraph = document.createElement("p");
    dmzIconParagraph.textContent = "DMZ";
    dmzIconDiv.appendChild(dmzIconParagraph);

    answerContainerDiv.appendChild(dmzIconDiv);

    const answerDiv = document.createElement("div");
    answerDiv.className = "answer";

    const answerParagraph = document.createElement("p");
    answerParagraph.innerText = answer
    answerDiv.appendChild(answerParagraph);

    answerContainerDiv.appendChild(answerDiv);

    // chat_zone div에 두 번째 자식 요소를 추가합니다.
    chatZoneDiv.appendChild(answerContainerDiv);
}

function send_Sinjoword() {
    const input_sentence = document.getElementById('input_sentence').value
    const data = { 'input_sentence': input_sentence };
    // 텍스트 초기화
    // AJAX 요청
    let xhr = new XMLHttpRequest();
    // 요청 보내는부분
    xhr.open('POST', 'http://127.0.0.1:8000/request', true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.onreadystatechange = function () {
        if (xhr.readyState === XMLHttpRequest.DONE) {
            // 임시로 버튼 1에 테스트해보기
            if (xhr.status === 200) {
                const result = JSON.parse(xhr.responseText);

                if(result.word.includes('신조어')){
                    let answer = `${result.word}\n\n ${result.mean}     \n\n${result.sentence}`
                    new_chat(input_sentence, answer) 
                }else{
                    console.log(result.word.includes('신조어')) 
                    let answer = `신조어)    ${result.word}   입니다.\n\n 신조어 뜻) \t   ${result.mean}     \n\n번역 문장) ${result.sentence}입니다. `
                    new_chat(input_sentence, answer)
                } 
            } else {
                console.error('Error:', xhr.status);
                document.getElementById('but1').innerText = 'Error occurred. Please try again.';
            }
        }
    };
    xhr.send(JSON.stringify(data));
    document.getElementById('input_sentence').value = ``;
}


function random_click(text){
    let word
    if (text == '1'){
        word = document.getElementById('but1').textContent
    }
    else if (text == '2'){
        word = document.getElementById('but2').textContent
    }
    else if (text == '3'){
        word = document.getElementById('but3').textContent
    }
    
    let xhr = new XMLHttpRequest();
    const data = { 'input_sentence': word };
    // 요청 보내는부분
    xhr.open('POST', 'http://127.0.0.1:8000/random', true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.onreadystatechange = function () {
        if (xhr.readyState === XMLHttpRequest.DONE) {
            // 임시로 버튼 1에 테스트해보기
            if (xhr.status === 200) {
                const result = JSON.parse(xhr.responseText);
                const answer = `해당 신조어의 뜻) ${result.mean}입니다.\n\n\n아래는 신조어를 보편적 표현으로 바꾼 것입니다. \n\n\n${result.sentence}`
                new_chat(word, answer)
            } else {
                console.error('Error:', xhr.status);
                document.getElementById('but1').innerText = 'Error occurred. Please try again.';
            }
        }
    };
    xhr.send(JSON.stringify(data));
}

