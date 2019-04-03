#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

def get_initial_context():
    context_string = input("Model prompt >>> ")
    while not context_string:
        print('Prompt should not be empty!')
        context_string = input("Model prompt >>> ")
    return context_string


def generate_text_from_context(context_string, **kwargs):
    nsamples, batch_size, output, context, enc, sess = [kwargs[k] for k in ['nsamples', 'batch_size', 'output', 'context', 'enc', 'sess']]

    context_tokens = enc.encode(context_string)
    generated = 0
    for _ in range(nsamples // batch_size):
        out = sess.run(output, feed_dict={
            context: [context_tokens for _ in range(batch_size)]
        })[:, len(context_tokens):]
        for i in range(batch_size):
            generated += 1
            text = enc.decode(out[i])
            print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            print(text)
 

def generate_next_sentence_from_context(context_string, **kwargs):
    output, context, enc, sess = [kwargs[k] for k in ['output','context', 'enc', 'sess']]

    context_tokens = enc.encode(context_string)
    next_sentence, end_of_text = '', False
    
    while not next_sentence:
        out = sess.run(output, feed_dict={context: [context_tokens]})[:, len(context_tokens):]
        out_string = enc.decode(out[0])
        if '.' in out_string:
            first_period_idx = out_string.find('.')
            next_sentence = out_string[:first_period_idx+1]
            if out_string[first_period_idx+1:first_period_idx+14] == '<|endoftext|>':
                next_sentence += ' THE END'
                end_of_text = True
    
    return next_sentence, end_of_text


def interact_model(
    model_name='117M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=40,
    branch=False,
    nbranches=4,
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=40 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 (default) means 40 words are considered at each step. 0 is a
     special setting meaning no restrictions. 40 generally is a good value.
    :branch=False : For each input, generate the samples sentence-by-sentence,
     providing nchoices many options each time, until a "<|endoftext|>" character is
     encountered, or the user manually terminates execution.
    :nbranches : If branch=True, nchoices gives the number of next-sentence options presented
     to the user. This argument is ignored if branch=False.
    """
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        if not branch:
            length = hparams.n_ctx // 2
        else:
            length = hparams.n_ctx_branch // 2 # 128 // 2 = 64 by default
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)
        
        kwargs = {'batch_size': batch_size, 'nsamples': nsamples, 'context': context, 'output': output, 'enc': enc, 'sess': sess}
        while True:
            context_string = get_initial_context()
            if branch:
                choice, choice_is_eot = -1, False
                while choice and not choice_is_eot:
                    num_generated_branches = 0
                    sentences, eots = [], []
                    print('Choices:')
                    print('(0) TERMINATE')
                    for i in range(nbranches):
                        sentence_string, end_of_text = generate_next_sentence_from_context(context_string, **kwargs)
                        sentences.append(sentence_string)
                        eots.append(end_of_text)
                        print('({}) {}'.format(i+1, sentence_string))

                    choice = input("Branch choice >>> ")
                    while not choice.isnumeric() or int(choice) not in range(nbranches+1):
                        print('Choice should be a number between 0 and {}'.format(nbranches+1))
                        choice = input("Branch choice >>> ")
                    
                    choice = int(choice)
                    if choice > 0:
                        choice_is_eot = eots[choice-1]
                        context_string += sentences[choice-1]
                        print('\n',context_string,'\n')

            else:
                generate_text_from_context(context_string, **kwargs)

if __name__ == '__main__':
    fire.Fire(interact_model)

